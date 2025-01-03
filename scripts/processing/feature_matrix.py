#scripts/processing/feature_matrix.py
# -*- coding:utf-8 -*-

#scripts/processing/feature_matrix.py
#-*- coding:utf-8 -*-

import logging
import numpy as np
import pandas as pd
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import DBSCAN
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import multiprocessing
from ..processing.identification import CompoundIdentifier
from ..processing.ms2_comparaison import add_ms2_scores
from ..utils.matching_utils import find_matches_window
from ..config.config import Config

class FeatureProcessor:
    """Classe pour le traitement et l'alignement des features."""
    
    def __init__(self):
        """Initialise le processeur de features avec la configuration."""
        self.config = Config.FEATURE_ALIGNMENT
        self.logger = logging.getLogger(__name__)

    def align_features_across_samples(
        self,
        samples_dir: Path
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Aligne les features entre plusieurs échantillons.
        
        Args:
            samples_dir: Répertoire contenant les sous-dossiers d'échantillons
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, Dict]: 
                (intensity_matrix, feature_df, raw_files)
        """
        print("\n🔄 Alignement des features entre échantillons...")
        
        all_peaks = []
        sample_names = []
        
        # Chargement des pics pour chaque échantillon
        for sample_dir in samples_dir.glob("*"):
            if sample_dir.is_dir():
                peaks_file = sample_dir / "ms1" / "common_peaks.parquet"
                if peaks_file.exists():
                    peaks = pd.read_parquet(peaks_file)
                    if not peaks.empty:
                        print(f"   ✓ Chargement de {sample_dir.name}: {len(peaks)} pics")
                        peaks = peaks.assign(
                            sample=sample_dir.name,
                            orig_rt=peaks['retention_time'],
                            orig_dt=peaks['drift_time']
                        )
                        all_peaks.append(peaks)
                        sample_names.append(sample_dir.name)
        
        if not all_peaks:
            raise ValueError("Aucun pic trouvé dans les échantillons")
        
        # Fusion des données de tous les échantillons
        df = pd.concat(all_peaks, ignore_index=True)
        print(f"   ✓ Total: {len(df)} pics à travers {len(sample_names)} échantillons")
        
        print("\n🎯 Clustering des features...")
        X = df[['mz', 'drift_time', 'retention_time']].to_numpy()
        median_mz = np.median(X[:, 0])
        
        X_scaled = np.column_stack([
            X[:, 0] / (median_mz * self.config.mz_ppm * 1e-6),
            X[:, 1] / self.config.dt_tolerance,
            X[:, 2] / self.config.rt_tolerance
        ])
        
        # Clustering avec DBSCAN
        clusters = DBSCAN(
            eps=self.config.dbscan_eps,
            min_samples=self.config.dbscan_min_samples,
            algorithm=self.config.algorithm,
            n_jobs=-1
        ).fit_predict(X_scaled)
        
        df['cluster'] = clusters
        non_noise_clusters = np.unique(clusters[clusters != -1])
        
        print("\n📊 Génération des features alignées...")
        cluster_groups = df[df['cluster'].isin(non_noise_clusters)].groupby('cluster')
        features = []
        intensities = {}
        
        for cluster_id, cluster_data in cluster_groups:
            max_intensity_idx = cluster_data['intensity'].idxmax()
            max_intensity_row = cluster_data.loc[max_intensity_idx]
            
            feature = {
                'mz': cluster_data['mz'].mean(),
                'retention_time': cluster_data['retention_time'].mean(),
                'drift_time': cluster_data['drift_time'].mean(),
                'intensity': max_intensity_row['intensity'],
                'source_sample': max_intensity_row['sample'],
                'source_rt': max_intensity_row['orig_rt'],
                'source_dt': max_intensity_row['orig_dt'],
                'n_samples': cluster_data['sample'].nunique(),
                'samples': ','.join(sorted(cluster_data['sample'].unique())),
                'feature_id': f"F{len(features) + 1:04d}"
            }
            
            if 'CCS' in cluster_data.columns:
                feature['CCS'] = cluster_data['CCS'].mean()
            
            features.append(feature)
            
            feature_name = f"{feature['feature_id']}_mz{feature['mz']:.4f}"
            sample_intensities = cluster_data.groupby('sample')['intensity'].max()
            intensities[feature_name] = sample_intensities
        
        feature_df = pd.DataFrame(features)
        print(f"   ✓ {len(feature_df)} features uniques détectées")
        
        # Création de la matrice d'intensités
        intensity_matrix = pd.DataFrame(intensities, index=sample_names).fillna(0)
        
        # Mapping des fichiers raw
        raw_files = {
            sample_dir.name: next(Path("data/input/samples").glob(f"{sample_dir.name}*.parquet"))
            for sample_dir in samples_dir.glob("*")
            if sample_dir.is_dir() and next(Path("data/input/samples").glob(f"{sample_dir.name}*.parquet"), None)
        }
        
        return intensity_matrix, feature_df, raw_files

    def process_features(
        self,
        feature_df: pd.DataFrame,
        raw_files: Dict,
        identifier: CompoundIdentifier
    ) -> pd.DataFrame:
        """
        Traite les features pour l'identification et l'extraction MS2.
        
        Args:
            feature_df: DataFrame des features
            raw_files: Dictionnaire des fichiers raw
            identifier: Instance de CompoundIdentifier
            
        Returns:
            pd.DataFrame: Features identifiées avec MS2
        """
        try:
            # Chargement des données MS2 par échantillon
            ms2_data = self._load_ms2_data(raw_files)
            
            # Extraction des spectres MS2
            feature_df = self._extract_ms2_spectra_optimized(feature_df, ms2_data)
            n_with_ms2 = sum(1 for x in feature_df['peaks_mz_ms2'] if len(x) > 0)
            print(f"{n_with_ms2}/{len(feature_df)} features avec spectres MS2")

            # Identification des composés
            matches = self._identify_features(feature_df, identifier)
            
            if not matches.empty:
                # Ajout des scores MS2
                add_ms2_scores(matches, identifier)
                
                # Tri final des colonnes
                cols = [col for col in matches.columns if col != 'confidence_level'] + ['confidence_level']
                matches = matches[cols]
                
            return matches

        except Exception as e:
            self.logger.error(f"Erreur lors du traitement des features : {str(e)}")
            raise

    def _load_ms2_data(self, raw_files: Dict) -> Dict[str, pd.DataFrame]:
        """Charge toutes les données MS2 en une fois."""
        ms2_data = {}
        for sample_name, raw_file in raw_files.items():
            raw_data = pd.read_parquet(raw_file)
            ms2_data[sample_name] = raw_data[raw_data['mslevel'].astype(int) == 2].copy()
        return ms2_data

    def _extract_ms2_spectra_optimized(
        self,
        feature_df: pd.DataFrame,
        ms2_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Extrait les spectres MS2 de manière optimisée par échantillon."""
        # Initialisation des colonnes MS2
        feature_df['peaks_mz_ms2'] = [[] for _ in range(len(feature_df))]
        feature_df['peaks_intensities_ms2'] = [[] for _ in range(len(feature_df))]

        # Grouper les features par échantillon source
        grouped_features = feature_df.groupby('source_sample')

        for sample_name, group in tqdm(grouped_features, desc="Extraction MS2"):
            if sample_name not in ms2_data:
                continue

            sample_ms2 = ms2_data[sample_name]
            
            for idx, feature in group.iterrows():
                rt_min = feature['source_rt'] - Config.MS2_EXTRACTION.rt_tolerance
                rt_max = feature['source_rt'] + Config.MS2_EXTRACTION.rt_tolerance
                dt_min = feature['source_dt'] - Config.MS2_EXTRACTION.dt_tolerance
                dt_max = feature['source_dt'] + Config.MS2_EXTRACTION.dt_tolerance

                # Extraire les spectres MS2 correspondants
                ms2_window = sample_ms2[
                    (sample_ms2['rt'].between(rt_min, rt_max)) &
                    (sample_ms2['dt'].between(dt_min, dt_max))
                ]

                if not ms2_window.empty:
                    # Traitement du spectre MS2
                    ms2_window['mz_rounded'] = ms2_window['mz'].round(
                        Config.MS2_EXTRACTION.mz_round_decimals
                    )
                    spectrum = ms2_window.groupby('mz_rounded')['intensity'].sum().reset_index()

                    max_intensity = spectrum['intensity'].max()
                    if max_intensity > 0:
                        spectrum['intensity_normalized'] = (
                            spectrum['intensity'] / max_intensity * 
                            Config.MS2_EXTRACTION.intensity_scale
                        ).round(0).astype(int)
                        
                        spectrum = spectrum.nlargest(
                            Config.MS2_EXTRACTION.max_peaks,
                            'intensity'
                        )
                        
                        feature_df.at[idx, 'peaks_mz_ms2'] = spectrum['mz_rounded'].tolist()
                        feature_df.at[idx, 'peaks_intensities_ms2'] = spectrum['intensity_normalized'].tolist()

        return feature_df

    def _identify_features(
        self,
        feature_df: pd.DataFrame,
        identifier: CompoundIdentifier
    ) -> pd.DataFrame:
        """Identifie les features dans la base de données."""
        db = identifier.db.copy()
        db = db.sort_values('mz')
        db_mz = db['mz'].values

        all_matches = []
        
        for idx, feature in tqdm(feature_df.iterrows(), total=len(feature_df),
                               desc="Identification"):
            mz_tolerance = feature['mz'] * Config.IDENTIFICATION.tolerances['mz_ppm'] * 1e-6
            mz_min, mz_max = feature['mz'] - mz_tolerance, feature['mz'] + mz_tolerance

            idx_start = np.searchsorted(db_mz, mz_min, side='left')
            idx_end = np.searchsorted(db_mz, mz_max, side='right')

            if idx_start == idx_end:
                continue

            feature_data = pd.DataFrame([{
                'mz': feature['mz'],
                'retention_time': feature['retention_time'],
                'drift_time': feature['drift_time'],
                'CCS': feature['CCS'],
                'intensity': feature['intensity']
            }])

            matches_for_feature = find_matches_window(
                feature_data,
                db.iloc[idx_start:idx_end]
            )
            
            if not matches_for_feature.empty:
                matches_for_feature['feature_idx'] = idx
                matches_for_feature['peaks_mz_ms2'] = [feature['peaks_mz_ms2']] * len(matches_for_feature)
                matches_for_feature['peaks_intensities_ms2'] = [feature['peaks_intensities_ms2']] * len(matches_for_feature)
                all_matches.append(matches_for_feature)

        if all_matches:
            return pd.concat(all_matches, ignore_index=True)
        return pd.DataFrame()

    def create_feature_matrix(
        self,
        input_dir: Path,
        output_dir: Path,
        identifier: CompoundIdentifier
    ) -> None:
        """
        Crée la matrice de features et les identifications.
        """
        try:
            # 1. Alignement des features
            matrix, feature_info, raw_files = self.align_features_across_samples(input_dir)
            
            # 2. Identification et MS2
            identifications = self.process_features(feature_info, raw_files, identifier)
            
            # 3. Sauvegardes
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Sauvegarder la matrice d'intensités
            matrix.to_parquet(output_dir / "feature_matrix.parquet")
            matrix.to_csv(output_dir / "feature_matrix.csv")
            
            # Si on a des identifications, créer les fichiers combinés
            if not identifications.empty:
                # Copier les colonnes MS2 de feature_info vers identifications si nécessaire
                ms2_columns = ['peaks_mz_ms2', 'peaks_intensities_ms2']
                for col in ms2_columns:
                    if col in feature_info.columns and col not in identifications.columns:
                        identifications[col] = feature_info.loc[identifications['feature_idx'], col].values

                # Création du résumé
                summary_df = pd.merge(
                    feature_info.reset_index().rename(columns={'index': 'feature_idx'}),
                    identifications,
                    on='feature_idx',
                    how='left'
                )
                
                # Sauvegarder en parquet
                summary_df.to_parquet(output_dir / "features_complete.parquet")
                
                # Pour le CSV, convertir les colonnes MS2 si présentes
                csv_df = summary_df.copy()
                ms2_cols_to_convert = [col for col in ms2_columns if col in csv_df.columns]
                
                for col in ms2_cols_to_convert:
                    csv_df[col] = csv_df[col].apply(
                        lambda x: ';'.join(map(str, x)) if isinstance(x, (list, np.ndarray)) else ''
                    )
                
                # Sauvegarde en CSV
                csv_df.to_csv(output_dir / "features_complete.csv", index=False)
            
            # 4. Affichage des statistiques
            self._print_feature_matrix_stats(matrix, feature_info, identifications)
            
        except Exception as e:
            print(f"❌ Erreur lors de la création de la matrice: {str(e)}")
            raise
                    
                    
                    
    def _print_feature_matrix_stats(
        self,
        matrix: pd.DataFrame,
        feature_info: pd.DataFrame,  # Ajout de ce paramètre manquant
        identifications: pd.DataFrame
    ) -> None:
        """
        Affiche les statistiques de la matrice de features.
        
        Args:
            matrix: Matrice d'intensités
            feature_info: DataFrame des informations sur les features
            identifications: DataFrame des identifications
        """
        print("\n✅ Création de la matrice des features terminée avec succès")
        print(f"   • {matrix.shape[1]} features")
        print(f"   • {matrix.shape[0]} échantillons")
        
        if not identifications.empty:
            print("\n📊 Distribution des niveaux de confiance:")
            for sample in sorted(matrix.index):
                print(f"\n   • {sample}:")
                
                # Trouver les features présentes dans cet échantillon
                sample_features = [idx for idx, name in enumerate(matrix.columns)
                                if matrix.loc[sample, name] > 0]
                
                # Filtrer les identifications pour ces features
                sample_identifications = identifications[
                    identifications['feature_idx'].isin(sample_features)
                ]

                if len(sample_identifications) > 0:
                    for level in sorted(sample_identifications['confidence_level'].unique()):
                        level_df = sample_identifications[
                            sample_identifications['confidence_level'] == level
                        ]
                        unique_molecules = level_df['match_name'].nunique()
                        print(f"      Niveau {level}: {unique_molecules} molécules uniques")
                else:
                    print("      Aucune identification")

def create_feature_matrix(
    input_dir: Path,
    output_dir: Path,
    identifier: CompoundIdentifier
) -> None:
    """
    Fonction wrapper pour créer la matrice de features.
    
    Args:
        input_dir: Répertoire d'entrée
        output_dir: Répertoire de sortie
        identifier: Instance de CompoundIdentifier
    """
    processor = FeatureProcessor()
    processor.create_feature_matrix(input_dir, output_dir, identifier)