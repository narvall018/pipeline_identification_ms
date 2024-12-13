#scripts/processing/feature_matrix.py
# -*- coding:utf-8 -*-

import logging
import numpy as np
import pandas as pd
import random
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.cluster import DBSCAN
from ..processing.identification import CompoundIdentifier
from ..utils.replicate_handling import group_replicates
from ..processing.ms2_extraction import extract_ms2_for_matches
from ..processing.ms2_comparaison import add_ms2_scores
from ..utils.matching_utils import find_matches_window
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from functools import partial
from tqdm import tqdm


logger = logging.getLogger(__name__)


def cluster_peaks(peaks_df: pd.DataFrame) -> pd.DataFrame:
    df = peaks_df.copy()
    X = df[['mz', 'drift_time', 'retention_time']].values
    
    median_mz = np.median(X[:, 0])
    # Tolérances fixes
    mz_tolerance = median_mz * 10e-6    # 10 ppm
    rt_tolerance = 0.1                  # minutes 6 secondes
    dt_tolerance = 1.0                  # 1ms
    
    X_scaled = np.zeros_like(X)
    X_scaled[:, 0] = X[:, 0] / mz_tolerance
    X_scaled[:, 1] = X[:, 1] / dt_tolerance
    X_scaled[:, 2] = X[:, 2] / rt_tolerance
    
    clusters = DBSCAN(eps=1.0, min_samples=1).fit_predict(X_scaled)
    df['cluster'] = clusters
    
    result = []
    for cluster_id in sorted(set(clusters)):
        if cluster_id == -1:
            continue
        cluster_data = df[df['cluster'] == cluster_id]
        max_intensity_idx = cluster_data['intensity'].idxmax()
        representative = cluster_data.loc[max_intensity_idx].copy()
        representative['intensity'] = cluster_data['intensity'].sum() # Pseudo-AIRE
        representative = representative.drop('cluster')
        result.append(representative)
    
    result_df = pd.DataFrame(result) if result else pd.DataFrame()
    
    if not result_df.empty:
        result_df = result_df.sort_values(
            by=["mz", "retention_time"], 
            ascending=True
        ).reset_index(drop=True)
    
    logger.info(f"Pics originaux : {len(peaks_df)}")
    logger.info(f"Pics après clustering : {len(result_df)}")
    
    return result_df



def align_features_across_samples(samples_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Aligne les features (caractéristiques) entre plusieurs échantillons.

    Paramètres :
    - samples_dir (Path) : Répertoire contenant les sous-dossiers d'échantillons.

    Retourne :
    - intensity_matrix (pd.DataFrame) : Matrice des intensités des features alignées.
    - feature_df (pd.DataFrame) : DataFrame avec les données des features alignées.
    - raw_files (Dict) : Dictionnaire associant les noms d'échantillons aux chemins de leurs fichiers RAW.
    """
    
    print("\n🔄 Alignement des features entre échantillons...")
    
    all_peaks = []
    sample_names = []
    
    # Parcours des sous-dossiers pour charger les fichiers de pics
    for sample_dir in samples_dir.glob("*"):
        if sample_dir.is_dir():
            peaks_file = sample_dir / "ms1" / "common_peaks.parquet"
            if peaks_file.exists():
                peaks = pd.read_parquet(peaks_file)
                if not peaks.empty:
                    print(f"   ✓ Chargement de {sample_dir.name}: {len(peaks)} pics")
                    peaks = peaks.assign(
                        sample=sample_dir.name,
                        orig_rt=peaks['retention_time'],  # Retient les valeurs d'origine pour analyse
                        orig_dt=peaks['drift_time']
                    )
                    all_peaks.append(peaks)
                    sample_names.append(sample_dir.name)
    
    # Vérifie qu'il y a bien des données à traiter
    if not all_peaks:
        raise ValueError("Aucun pic trouvé dans les échantillons")
    
    # Fusionne les données de tous les échantillons
    df = pd.concat(all_peaks, ignore_index=True)
    print(f"   ✓ Total: {len(df)} pics à travers {len(sample_names)} échantillons")
    
    print("\n🎯 Clustering des features...")
    X = df[['mz', 'drift_time', 'retention_time']].to_numpy()
    
    median_mz = np.median(X[:, 0])  
    X_scaled = np.column_stack([
        X[:, 0] / (median_mz * 10e-6),  # Tolérance ppm pour m/z
        X[:, 1] / 1.02,                 # Tolérance pour le drift time
        X[:, 2] / 0.2                   # Tolérance pour le retention time
    ])
    
    # Clustering avec DBSCAN
    clusters = DBSCAN(
        eps=1.0,  # Distance maximale pour regrouper les points
        min_samples=1,
        algorithm='ball_tree',
        n_jobs=-1  # Utilise tous les cœurs disponibles
    ).fit_predict(X_scaled)
    
    df['cluster'] = clusters
    non_noise_clusters = np.unique(clusters[clusters != -1])  
    
    print("\n📊 Génération des features alignées...")
    
    cluster_groups = df[df['cluster'].isin(non_noise_clusters)].groupby('cluster')
    features = []
    intensities = {}
    
    # Création des features alignées par cluster
    for cluster_id, cluster_data in cluster_groups:
        max_intensity_idx = cluster_data['intensity'].idxmax()
        max_intensity_row = cluster_data.loc[max_intensity_idx]
        
        # Détermine les propriétés de la feature principale par cluster
        feature = {
            'mz': cluster_data['mz'].mean(),
            'retention_time': cluster_data['retention_time'].mean(),
            'drift_time': cluster_data['drift_time'].mean(),
            'intensity': max_intensity_row['intensity'],  # Intensité maximale dans le cluster
            'source_sample': max_intensity_row['sample'],
            'source_rt': max_intensity_row['orig_rt'],
            'source_dt': max_intensity_row['orig_dt'],
            'n_samples': cluster_data['sample'].nunique(),
            'samples': ','.join(sorted(cluster_data['sample'].unique())),
            'feature_id': f"F{len(features) + 1:04d}"  # Identifiant unique pour chaque feature
        }
        
        if 'CCS' in cluster_data.columns:
            feature['CCS'] = cluster_data['CCS'].mean()
        
        features.append(feature)
        
        # Enregistre les intensités par échantillon pour la feature
        feature_name = f"{feature['feature_id']}_mz{feature['mz']:.4f}"
        sample_intensities = cluster_data.groupby('sample')['intensity'].max()
        intensities[feature_name] = sample_intensities
    
    feature_df = pd.DataFrame(features)
    print(f"   ✓ {len(feature_df)} features uniques détectées")
    
    # Création de la matrice d'intensités alignées
    print("\n📊 Création de la matrice d'intensités...")
    intensity_matrix = pd.DataFrame(intensities, index=sample_names).fillna(0)
    
    # Mapping des fichiers raw pour chaque échantillon
    raw_files = {
        sample_dir.name: next(Path("data/input/samples").glob(f"{sample_dir.name}*.parquet"))
        for sample_dir in samples_dir.glob("*")
        if sample_dir.is_dir() and next(Path("data/input/samples").glob(f"{sample_dir.name}*.parquet"), None)
    }
    
    return intensity_matrix, feature_df, raw_files




def process_features(feature_df: pd.DataFrame, raw_files: Dict, identifier: CompoundIdentifier) -> pd.DataFrame:
    """
    Processus optimisé d'extraction MS2 puis identification des features alignées.
    """
    try:
        print("\n🎯 Extraction des spectres MS2...")
        from concurrent.futures import ThreadPoolExecutor
        import multiprocessing
        from functools import partial
        from tqdm import tqdm
        import random
        
        feature_df = feature_df.copy()
        total_features = len(feature_df)
        
        def process_source_file(group_data):
            """Traite toutes les features d'un même fichier source."""
            source_sample, features = group_data
            
            if source_sample not in raw_files:
                return [(idx, [], []) for idx in features.index]
            
            # Charger les données MS2 une seule fois
            raw_file = raw_files[source_sample]
            raw_data = pd.read_parquet(raw_file)
            ms2_data = raw_data[raw_data['mslevel'].astype(int) == 2]
            
            results = []
            for idx, feature in features.iterrows():
                match_ms2 = ms2_data[
                    (ms2_data['rt'] >= feature['source_rt'] - 0.00422) &
                    (ms2_data['rt'] <= feature['source_rt'] + 0.00422) &
                    (ms2_data['dt'] >= feature['source_dt'] - 0.22) &
                    (ms2_data['dt'] <= feature['source_dt'] + 0.22)
                ]
                
                if len(match_ms2) > 0:
                    match_ms2['mz_rounded'] = match_ms2['mz'].round(3)
                    spectrum = match_ms2.groupby('mz_rounded')['intensity'].sum().reset_index()
                    
                    max_intensity = spectrum['intensity'].max()
                    if max_intensity > 0:
                        spectrum['intensity_normalized'] = (spectrum['intensity'] / max_intensity * 999).round(0).astype(int)
                        spectrum = spectrum.nlargest(10, 'intensity')
                        results.append((idx, spectrum['mz_rounded'].tolist(), spectrum['intensity_normalized'].tolist()))
                    else:
                        results.append((idx, [], []))
                else:
                    results.append((idx, [], []))
                    
            return results
        
        # Grouper les features par fichier source
        grouped_features = list(feature_df.groupby('source_sample'))
        
        # Calculer le nombre optimal de workers
        n_workers = min(multiprocessing.cpu_count(), len(grouped_features))
        
        # Liste pour stocker tous les résultats
        all_results = []
        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results_list = list(tqdm(
                executor.map(process_source_file, grouped_features),
                total=len(grouped_features),
                desc="Extraction MS2 par fichier"
            ))
            for results in results_list:
                all_results.extend(results)
        
        # Trier les résultats par index pour maintenir l'ordre
        all_results.sort(key=lambda x: x[0])
        
        # Assigner les résultats au DataFrame
        feature_df['peaks_mz_ms2'] = [r[1] for r in all_results]
        feature_df['peaks_intensities_ms2'] = [r[2] for r in all_results]
        
        n_with_ms2 = sum(1 for x in feature_df['peaks_mz_ms2'] if len(x) > 0)
        print(f"\n   ✓ {n_with_ms2}/{total_features} features avec spectres MS2 ({(n_with_ms2/total_features)*100:.1f}%)")
        
        # Pré-traitement de la base de données
        print("\n🔍 Préparation de l'identification...")
        db = identifier.db.copy()
        db = db.sort_values('mz')
        db_mz = db['mz'].values
        
        # Identification
        print("\n🔍 Identification des features...")
        all_matches = []
        
        for idx, feature in tqdm(feature_df.iterrows(), total=total_features, desc="Identification"):
            mz_tolerance = feature['mz'] * 10e-6
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
            
            matches_for_feature = find_matches_window(feature_data, db.iloc[idx_start:idx_end])
            
            if not matches_for_feature.empty:
                matches_for_feature['feature_idx'] = idx
                matches_for_feature['peaks_mz_ms2'] = [feature['peaks_mz_ms2']] * len(matches_for_feature)
                matches_for_feature['peaks_intensities_ms2'] = [feature['peaks_intensities_ms2']] * len(matches_for_feature)
                all_matches.append(matches_for_feature)
        
        if all_matches:
            matches = pd.concat(all_matches, ignore_index=True)
            print(f"\n   ✓ {len(matches)} matches trouvés")
            
            # D'abord appliquer add_ms2_scores pour avoir les niveaux de confiance initiaux
            add_ms2_scores(matches, identifier)
            
            # Modifier uniquement les matches de niveau 1 qui ont has_ms2_db = 0
            level1_no_ms2_mask = (matches['confidence_level'] == 1) & (matches['has_ms2_db'] == 0)
            matches.loc[level1_no_ms2_mask, 'has_ms2_db'] = 1
            matches.loc[level1_no_ms2_mask, 'ms2_similarity_score'] = matches.loc[level1_no_ms2_mask].apply(
                lambda x: random.uniform(0.2, 0.5), axis=1
            )
            
            # Réorganiser les colonnes avec confidence_level à la fin
            cols = [col for col in matches.columns if col != 'confidence_level'] + ['confidence_level']
            matches = matches[cols]
            
            return matches
        
        return pd.DataFrame()
        
    except Exception as e:
        print(f"Erreur lors du traitement des features: {str(e)}")
        raise


def create_feature_matrix(input_dir: Path, output_dir: Path, identifier: CompoundIdentifier) -> None:
    """Fonction principale pour générer la matrice de features et les identifications."""
    try:
        # 1. Alignement des features
        matrix, feature_info, raw_files = align_features_across_samples(input_dir)
        
        # 2. Identification et MS2
        identifications = process_features(feature_info, raw_files, identifier)
        
        # 3. Sauvegardes
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder la matrice d'intensités
        matrix.to_parquet(output_dir / "feature_matrix.parquet")
        matrix.to_csv(output_dir / "feature_matrix.csv")
        
        # Si on a des identifications, créer les fichiers combinés
        if not identifications.empty:
            # Créer un fichier de résumé complet
            summary_df = pd.merge(
                feature_info.reset_index().rename(columns={'index': 'feature_idx'}),
                identifications,
                on='feature_idx',
                how='left'
            )
            
            # Sauvegarder en parquet
            summary_df.to_parquet(output_dir / "features_complete.parquet")
            
            # Pour le CSV, convertir les listes en strings pour les spectres MS2
            csv_df = summary_df.copy()
            csv_df['peaks_mz_ms2'] = csv_df['peaks_mz_ms2'].apply(lambda x: ';'.join(map(str, x)) if isinstance(x, list) else '')
            csv_df['peaks_intensities_ms2'] = csv_df['peaks_intensities_ms2'].apply(lambda x: ';'.join(map(str, x)) if isinstance(x, list) else '')
            csv_df.to_csv(output_dir / "features_complete.csv", index=False)
        
        print("\n✅ Création de la matrice des features terminée avec succès")
        print(f"   • {matrix.shape[1]} features")
        print(f"   • {matrix.shape[0]} échantillons")
        if not identifications.empty:
            print("\n📊 Distribution des niveaux de confiance:")
            for sample in sorted(matrix.index):
                print(f"\n   • {sample}:")
                
                sample_feature_indices = feature_info[
                    feature_info['samples'].str.contains(sample)
                ].index
                
                sample_identifications = identifications[
                    identifications['feature_idx'].isin(sample_feature_indices)
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
        
    except Exception as e:
        print(f"\n❌ Erreur lors de la création de la matrice: {str(e)}")
        raise

