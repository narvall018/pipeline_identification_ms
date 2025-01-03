#scripts/processing/blank_processing.py
#-*- coding:utf-8 -*-

from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from ..config.config import Config
from .peak_detection import PeakDetector

class BlankProcessor:
    """Classe responsable du traitement des blancs et de la soustraction."""
    
    def __init__(self):
        """Initialise le processeur de blanks avec la configuration."""
        self.blank_config = Config.BLANK_SUBTRACTION
        self.peak_detector = PeakDetector()

    def process_blank_file(
        self,
        file_path: Union[str, Path],
        data_type: str = 'blanks'
    ) -> Optional[pd.DataFrame]:
        """
        Traite un fichier blank individuel.
        
        Args:
            file_path: Chemin vers le fichier blank
            data_type: Type de données ('blanks' par défaut)
            
        Returns:
            Optional[pd.DataFrame]: Pics du blank traités ou None si échec
        """
        try:
            data = pd.read_parquet(file_path)
            return self.peak_detector.process_sample(data)
        except Exception as e:
            print(f"❌ Erreur lors du traitement du fichier {file_path}: {str(e)}")
            return None

    def process_blank_with_replicates(
        self,
        blank_name: str, 
        replicate_files: List[Path],
        output_dir: Path
    ) -> pd.DataFrame:
        """
        Traite les réplicats d'un blank.
        
        Args:
            blank_name: Nom du blank
            replicate_files: Liste des fichiers réplicats
            output_dir: Répertoire de sortie
            
        Returns:
            pd.DataFrame: Pics communs aux réplicats
        """
        print(f"\n{'='*80}")
        print(f"TRAITEMENT DU BLANK {blank_name}")
        print(f"{'='*80}")

        all_peaks = {}
        for rep_file in replicate_files:
            # Renommer le fichier pour enlever "_replicate_"
            rep_name = rep_file.stem.replace('_replicate_', '_')
            
            peaks = self.process_blank_file(rep_file)
            if peaks is not None and not peaks.empty:
                all_peaks[rep_name] = peaks

        if not all_peaks:
            print("   Aucun pic détecté dans les réplicats.")
            return pd.DataFrame()
                
        print("\nPICS PAR RÉPLICAT:")
        for rep, df in all_peaks.items():
            print(f"   {rep}: {len(df)} pics")

        if len(all_peaks) == 1:
            unique_df = list(all_peaks.values())[0]
            print(f"\n   1 seul réplicat traité.")
            print(f"   Pics finaux : {len(unique_df)}")
            return unique_df

        min_required = 2 if len(replicate_files) == 3 else len(replicate_files)
        print(f"\n   ℹ️ Critère: {min_required}/{len(replicate_files)} réplicats requis")
        combined_peaks = self.cluster_blank_replicates(all_peaks, min_required)
        print(f"\n   Pics finaux après convergence : {len(combined_peaks)}")

        return combined_peaks

    def cluster_blank_replicates(
        self,
        peaks_dict: Dict[str, pd.DataFrame],
        min_required: int
    ) -> pd.DataFrame:
        """
        Cluster les pics entre réplicats de blanks.
        
        Args:
            peaks_dict: Dictionnaire des pics par réplicat
            min_required: Nombre minimum de réplicats requis
            
        Returns:
            pd.DataFrame: Pics communs
        """
        # Indentation correcte de tout le bloc suivant
        all_peaks = pd.concat(
            [peaks.assign(replicate=name) for name, peaks in peaks_dict.items()],
            ignore_index=True
        )
        
        if len(all_peaks) == 0:
            return pd.DataFrame()
        
        # Extraction des coordonnées pour le clustering
        X = all_peaks[['mz', 'drift_time', 'retention_time']].to_numpy()
        median_mz = np.median(X[:, 0])
        mz_tolerance = median_mz * Config.BLANK_REPLICATE.mz_ppm * 1e-6
        
        # Normalisation des dimensions pour le clustering
        X_scaled = np.column_stack([
            X[:, 0] / mz_tolerance,  # Normalisation m/z avec ppm
            X[:, 1] / Config.BLANK_REPLICATE.dt_tolerance,  # Normalisation drift time
            X[:, 2] / Config.BLANK_REPLICATE.rt_tolerance   # Normalisation retention time
        ])
        
        # Clustering DBSCAN avec les paramètres de la configuration
        clusters = DBSCAN(
            eps=Config.BLANK_REPLICATE.dbscan_eps,
            min_samples=min_required,
            algorithm=Config.BLANK_REPLICATE.algorithm,
            n_jobs=-1  # Utilise tous les coeurs disponibles
        ).fit_predict(X_scaled)
        
        all_peaks['cluster'] = clusters
        valid_clusters = all_peaks[clusters != -1].groupby('cluster')
        
        result = []
        for _, cluster_data in valid_clusters:
            # Vérifie si le cluster est présent dans suffisamment de réplicats
            n_replicates = cluster_data['replicate'].nunique()
            if n_replicates >= min_required:
                # Prend le pic le plus intense comme représentant du cluster
                max_intensity_idx = cluster_data['intensity'].idxmax()
                representative = cluster_data.loc[max_intensity_idx].copy()
                representative['n_replicates'] = n_replicates
                result.append(representative)
        
        # Création du DataFrame final
        result_df = pd.DataFrame(result) if result else pd.DataFrame()
        
        # Tri par intensité si des résultats existent
        if not result_df.empty:
            result_df = result_df.sort_values('intensity', ascending=False)
        
        return result_df

    def subtract_blank_peaks(
        self,
        sample_peaks: pd.DataFrame,
        blank_peaks: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Soustrait les pics du blank des pics de l'échantillon.
        
        Args:
            sample_peaks: Pics de l'échantillon
            blank_peaks: Pics du blank
            
        Returns:
            pd.DataFrame: Pics après soustraction du blank
        """
        if blank_peaks.empty or sample_peaks.empty:
            return sample_peaks

        combined = pd.concat(
            [sample_peaks.assign(is_sample=True),
             blank_peaks.assign(is_sample=False)],
            ignore_index=True
        )
        
        if combined.empty:
            return sample_peaks

        X = combined[['mz', 'drift_time', 'retention_time']].values
        median_mz = np.median(X[:, 0])
        mz_tolerance = median_mz * self.blank_config.mz_ppm * 1e-6

        X_scaled = np.column_stack([
            X[:, 0] / mz_tolerance,
            X[:, 1] / self.blank_config.dt_tolerance,
            X[:, 2] / self.blank_config.rt_tolerance
        ])

        clusters = DBSCAN(
            eps=self.blank_config.dbscan_eps,
            min_samples=self.blank_config.dbscan_min_samples,
            algorithm='ball_tree',
            n_jobs=-1
        ).fit_predict(X_scaled)
        
        combined['cluster'] = clusters

        # Un cluster est considéré comme blank si au moins 50% des pics sont des blanks
        blank_clusters = set()
        for cluster_id in combined[combined['cluster'] != -1]['cluster'].unique():
            cluster_data = combined[combined['cluster'] == cluster_id]
            if sum(~cluster_data['is_sample']) / len(cluster_data) >= self.blank_config.cluster_ratio:
                blank_clusters.add(cluster_id)

        # Filtrer les pics
        clean_peaks = combined[
            combined['is_sample'] & 
            (~combined['cluster'].isin(blank_clusters) | (combined['cluster'] == -1))
        ].copy()

        clean_peaks = clean_peaks.drop(['is_sample', 'cluster'], axis=1)
        
        print(f"{len(clean_peaks)} pics après soustraction du blank")
        return clean_peaks