#scripts/processing/replicate_processing.py
#-*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import DBSCAN
from ..config.config import Config
from .peak_detection import PeakDetector

class ReplicateProcessor:
    """Classe responsable du traitement des réplicats d'échantillons."""
    
    def __init__(self):
        """Initialise le processeur de réplicats avec la configuration."""
        self.replicate_config = Config.REPLICATE
        self.peak_detector = PeakDetector()
        
    def process_replicates(
        self,
        replicate_files: List[Path]
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, int]]:
        """
        Traite les réplicats d'un échantillon.
        
        Args:
            replicate_files: Liste des fichiers réplicats
            
        Returns:
            Tuple[Dict[str, pd.DataFrame], Dict[str, int]]: 
                (peaks_dict, initial_peaks)
        """
        all_peaks = {}
        initial_peak_counts = {}
        
        for rep_file in replicate_files:
            try:
                # Process each replicate
                data = pd.read_parquet(rep_file)
                processed_data = self.peak_detector.prepare_data(data)
                peaks = self.peak_detector.detect_peaks(processed_data)
                
                # Stocker le nombre de pics avant clustering
                initial_peak_counts[rep_file.stem] = len(peaks)
                
                # Clustering uniquement
                clustered_peaks = self.peak_detector.cluster_peaks(peaks)
                all_peaks[rep_file.stem] = clustered_peaks
                
                print(f"   ✓ {rep_file.stem}:")
                print(f"      - Pics initiaux: {initial_peak_counts[rep_file.stem]}")
                print(f"      - Pics après clustering: {len(clustered_peaks)}")
                
            except Exception as e:
                print(f"   ✗ Erreur avec {rep_file.stem}: {str(e)}")
        
        return all_peaks, initial_peak_counts

    def cluster_replicates(
        self,
        peaks_dict: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Cluster les pics entre réplicats.
        
        Args:
            peaks_dict: Dictionnaire des pics par réplicat
            
        Returns:
            pd.DataFrame: Pics communs entre réplicats
        """
        if len(peaks_dict) == 1:
            return list(peaks_dict.values())[0]
        
        # Combiner tous les réplicats efficacement
        all_peaks = pd.concat(
            [peaks.assign(replicate=name) for name, peaks in peaks_dict.items()],
            ignore_index=True
        )
        
        if len(all_peaks) == 0:
            return pd.DataFrame()
        
        # Configuration clustering
        total_replicates = len(peaks_dict)
        min_required = 2 if total_replicates == 3 else total_replicates
        
        # Calcul des tolérances et normalisation
        X = all_peaks[['mz', 'drift_time', 'retention_time']].to_numpy()
        median_mz = np.median(X[:, 0])
        mz_tolerance = median_mz * self.replicate_config.mz_ppm * 1e-6
        
        X_scaled = np.column_stack([
            X[:, 0] / mz_tolerance,
            X[:, 1] / self.replicate_config.dt_tolerance,
            X[:, 2] / self.replicate_config.rt_tolerance
        ])
        
        # Clustering optimisé
        clusters = DBSCAN(
            eps=self.replicate_config.dbscan_eps,
            min_samples=min_required,
            algorithm=self.replicate_config.algorithm,
            n_jobs=-1
        ).fit_predict(X_scaled)
        
        # Application du masque et groupement
        all_peaks['cluster'] = clusters
        valid_clusters = all_peaks[clusters != -1].groupby('cluster')
        
        # Construction du résultat
        result = []
        for _, cluster_data in valid_clusters:
            n_replicates = cluster_data['replicate'].nunique()
            
            if ((total_replicates == 2 and n_replicates == 2) or
                (total_replicates == 3 and n_replicates >= 2)):
                
                representative = {
                    'mz': cluster_data['mz'].max(),
                    'drift_time': cluster_data['drift_time'].max(),
                    'retention_time': cluster_data['retention_time'].max(),
                    'intensity': cluster_data['intensity'].max(),
                    'n_replicates': n_replicates
                }
                
                # Gestion CCS si présent
                if 'CCS' in cluster_data.columns:
                    representative['CCS'] = cluster_data['CCS'].mean()
                
                # Autres colonnes
                meta_cols = [col for col in cluster_data.columns if col not in 
                            ['mz', 'drift_time', 'retention_time', 'intensity', 'CCS', 'cluster', 'replicate']]
                for col in meta_cols:
                    representative[col] = cluster_data[col].iloc[0]
                
                result.append(representative)
        
        # Création et tri du DataFrame final
        result_df = pd.DataFrame(result) if result else pd.DataFrame()
        if not result_df.empty:
            result_df = result_df.sort_values('intensity', ascending=False)
        
        return result_df

    def process_sample_with_replicates(
        self,
        sample_name: str,
        replicate_files: List[Path],
        output_dir: Path
    ) -> pd.DataFrame:
        """
        Traite un échantillon avec ses réplicats.
        
        Args:
            sample_name: Nom de l'échantillon
            replicate_files: Liste des fichiers réplicats
            output_dir: Répertoire de sortie
            
        Returns:
            pd.DataFrame: Pics finaux traités
        """
        try:
            print(f"\n{'='*80}")
            print(f"Traitement de {sample_name}")
            print(f"{'='*80}")
            
            print(f"\n🔍 Traitement des réplicats ({len(replicate_files)} fichiers)...")
            
            # Traitement des réplicats
            peaks_data = self.process_replicates(replicate_files)
            peaks_dict, initial_peaks = peaks_data
            
            if not peaks_dict:
                print("   ✗ Aucun pic trouvé")
                return pd.DataFrame()
                
            # Clustering ou pics directs selon le nombre de réplicats
            if len(replicate_files) > 1:
                print(f"\n🔄 Clustering des pics entre réplicats...")
                final_peaks = self.cluster_replicates(peaks_dict)
                if not final_peaks.empty:
                    print(f"   ✓ {len(final_peaks)} pics communs trouvés")
                else:
                    print("   ✗ Aucun pic commun trouvé")
                    return pd.DataFrame()
            else:
                print("\n🔄 Traitement réplicat unique...")
                final_peaks = list(peaks_dict.values())[0]
                print(f"   ✓ {len(final_peaks)} pics trouvés")
            
            # Sauvegarde des pics intermédiaires
            output_dir = output_dir / sample_name / "ms1"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / "peaks_before_blank.parquet"
            final_peaks.to_parquet(output_file)
            
            # Résumé final
            print(f"\n✨ Traitement complet pour {sample_name}")
            if len(replicate_files) > 1:
                for rep_name in peaks_dict:
                    print(f"   - {rep_name}:")
                    print(f"      • Pics initiaux: {initial_peaks[rep_name]}")
                    print(f"      • Pics après clustering: {len(peaks_dict[rep_name])}")
                print(f"   - Pics communs: {len(final_peaks)}")
            else:
                rep_name = list(peaks_dict.keys())[0]
                print(f"   - Pics initiaux: {initial_peaks[rep_name]}")
                print(f"   - Pics après clustering: {len(final_peaks)}")
            
            return final_peaks
            
        except Exception as e:
            print(f"❌ Erreur lors du traitement de {sample_name}: {str(e)}")
            return pd.DataFrame()