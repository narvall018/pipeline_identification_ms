# -*- coding:utf-8 -*-

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.cluster import DBSCAN
from ..processing.identification import CompoundIdentifier
from ..utils.replicate_handling import group_replicates
from ..processing.ms2_extraction import extract_ms2_for_matches
from ..processing.ms2_comparaison import add_ms2_scores
from ..utils.matching_utils import find_matches_window

logger = logging.getLogger(__name__)

def align_features_across_samples(samples_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Aligne les features à travers les échantillons après filtration.
    """
    print("\n🔄 Alignement des features entre échantillons...")
    
    # Charger les pics filtrés/nettoyés de chaque échantillon
    all_peaks = []
    sample_names = []
    
    for sample_dir in samples_dir.glob("*"):
        if sample_dir.is_dir():
            peaks_file = sample_dir / "ms1" / "common_peaks.parquet"
            if peaks_file.exists():
                peaks = pd.read_parquet(peaks_file)
                if not peaks.empty:
                    print(f"   ✓ Chargement de {sample_dir.name}: {len(peaks)} pics")
                    peaks['sample'] = sample_dir.name
                    # Garder RT/DT originaux pour l'extraction MS2
                    peaks['orig_rt'] = peaks['retention_time']
                    peaks['orig_dt'] = peaks['drift_time']
                    all_peaks.append(peaks)
                    sample_names.append(sample_dir.name)
    
    if not all_peaks:
        raise ValueError("Aucun pic trouvé dans les échantillons")
    
    df = pd.concat(all_peaks, ignore_index=True)
    print(f"   ✓ Total: {len(df)} pics à travers {len(sample_names)} échantillons")
    
    print("\n🎯 Clustering des features...")
    X = df[['mz', 'drift_time', 'retention_time']].values
    
    # Mêmes tolérances que peak_detection
    mz_tolerance = np.median(X[:, 0]) * 1e-4  # 0.1 ppm
    dt_tolerance = np.median(X[:, 1]) * 0.10   # 10%
    rt_tolerance = 0.20                        # 0.2 min
    
    # Normalisation
    X_scaled = np.zeros_like(X)
    X_scaled[:, 0] = X[:, 0] / mz_tolerance
    X_scaled[:, 1] = X[:, 1] / dt_tolerance
    X_scaled[:, 2] = X[:, 2] / rt_tolerance
    
    # Clustering
    clusters = DBSCAN(eps=1.0, min_samples=1).fit_predict(X_scaled)
    df['cluster'] = clusters
    
    # Traitement des features alignées
    print("\n📊 Génération des features alignées...")
    feature_info = []
    
    # Obtenir une table de mapping des fichiers raw
    raw_files = {
        sample_dir.name: list(Path("data/input/samples").glob(f"{sample_dir.name}*.parquet"))[0]
        for sample_dir in samples_dir.glob("*")
        if sample_dir.is_dir() and list(Path("data/input/samples").glob(f"{sample_dir.name}*.parquet"))
    }
    
    for cluster_id in sorted(set(clusters)):
        if cluster_id == -1:
            continue
            
        cluster_data = df[df['cluster'] == cluster_id]
        # Trouver la version la plus intense
        max_index = cluster_data['intensity'].idxmax()
        max_intensity_row = cluster_data.loc[max_index]
        
        # Créer la feature avec moyennes
        feature = {
            'mz': cluster_data['mz'].mean(),
            'retention_time': cluster_data['retention_time'].mean(),
            'drift_time': cluster_data['drift_time'].mean(),
            'CCS': cluster_data['CCS'].mean(),
            'intensity': max_intensity_row['intensity'],
            'source_sample': max_intensity_row['sample'],
            'source_rt': max_intensity_row['orig_rt'],      
            'source_dt': max_intensity_row['orig_dt'],      
            'n_samples': cluster_data['sample'].nunique(),
            'samples': ','.join(sorted(cluster_data['sample'].unique())),
            'feature_id': f"F{len(feature_info) + 1:04d}"  # Ajout d'un ID unique
        }
        
        feature_info.append(feature)
    
    feature_df = pd.DataFrame(feature_info)
    print(f"   ✓ {len(feature_df)} features uniques détectées")
    
    # Création de la matrice d'intensités
    print("\n📊 Création de la matrice d'intensités...")
    intensity_matrix = pd.DataFrame(index=sample_names)
    
    for idx, feature in feature_df.iterrows():
        feature_name = f"{feature['feature_id']}_mz{feature['mz']:.4f}"
        cluster_data = df[df['cluster'] == idx]
        intensity_matrix[feature_name] = cluster_data.groupby('sample')['intensity'].max()
    
    return intensity_matrix, feature_df, raw_files



def process_features(feature_df: pd.DataFrame, raw_files: Dict, identifier: CompoundIdentifier) -> pd.DataFrame:
    """
    Processus optimisé d'extraction MS2 puis identification des features alignées.
    """
    print("\n🎯 Extraction des spectres MS2...")
    feature_df = feature_df.copy()
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial
    from tqdm import tqdm

    def extract_ms2_single(feature, raw_files):
        """Extraction MS2 pour une feature unique"""
        if feature['source_sample'] not in raw_files:
            return [], []
            
        raw_file = raw_files[feature['source_sample']]
        temp_df = pd.DataFrame([{
            'peak_rt': feature['source_rt'],
            'peak_dt': feature['source_dt']
        }])
        
        temp_df = extract_ms2_for_matches(temp_df, raw_file, "temp", silent=True)
        
        if temp_df is not None and not temp_df.empty:
            return (temp_df.iloc[0].get('peaks_mz_ms2', []),
                   temp_df.iloc[0].get('peaks_intensities_ms2', []))
        return [], []

    # Extraction MS2 parallèle
    n_with_ms2 = 0
    total_features = len(feature_df)
    
    # Utiliser le multithreading pour accélérer l'extraction
    with ThreadPoolExecutor(max_workers=4) as executor:
        extract_func = partial(extract_ms2_single, raw_files=raw_files)
        results = list(tqdm(
            executor.map(extract_func, [feature_df.iloc[i] for i in range(total_features)]),
            total=total_features,
            desc="Extraction MS2"
        ))
    
    # Attribution des résultats
    peaks_mz_ms2_list, peaks_intensities_ms2_list = zip(*results)
    feature_df['peaks_mz_ms2'] = peaks_mz_ms2_list
    feature_df['peaks_intensities_ms2'] = peaks_intensities_ms2_list
    
    n_with_ms2 = sum(1 for x in peaks_mz_ms2_list if len(x) > 0)
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
        # Filtre rapide sur m/z
        mz_tolerance = feature['mz'] * 10e-6
        mz_min, mz_max = feature['mz'] - mz_tolerance, feature['mz'] + mz_tolerance
        
        # Recherche binaire rapide
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
            # Répliquer les données MS2 pour chaque match
            matches_for_feature['feature_idx'] = idx
            matches_for_feature['peaks_mz_ms2'] = [feature['peaks_mz_ms2']] * len(matches_for_feature)
            matches_for_feature['peaks_intensities_ms2'] = [feature['peaks_intensities_ms2']] * len(matches_for_feature)
            all_matches.append(matches_for_feature)
    
    if all_matches:
        matches = pd.concat(all_matches, ignore_index=True)
        print(f"\n   ✓ {len(matches)} matches trouvés")
        
        print("\n🎯 Calcul des scores MS2...")
        add_ms2_scores(matches, identifier)
        
        print(f"   ✓ {len(matches)} identifications finales")
        return matches
    
    return pd.DataFrame()

def create_feature_matrix(input_dir: Path, output_dir: Path, identifier: CompoundIdentifier) -> None:
    """
    Fonction principale pour générer la matrice de features et les identifications.
    """
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
        
        # Sauvegarder les infos des features
        feature_info.to_parquet(output_dir / "feature_info.parquet")
        
        # Sauvegarder les identifications
        if not identifications.empty:
            identifications.to_parquet(output_dir / "feature_identifications.parquet")
            
            # Créer un fichier de résumé complet
            summary_df = pd.merge(
                feature_info.reset_index().rename(columns={'index': 'feature_idx'}),
                identifications,
                on='feature_idx',
                how='left'
            )
            
            # Sauvegarder en formats Parquet et CSV
            summary_df.to_parquet(output_dir / "complete_results.parquet")
            
            # Pour le CSV, convertir les listes en strings
            csv_df = summary_df.copy()
            csv_df['peaks_mz_ms2'] = csv_df['peaks_mz_ms2'].apply(lambda x: ';'.join(map(str, x)) if isinstance(x, list) else '')
            csv_df['peaks_intensities_ms2'] = csv_df['peaks_intensities_ms2'].apply(lambda x: ';'.join(map(str, x)) if isinstance(x, list) else '')
            csv_df.to_csv(output_dir / "complete_results.csv", index=False)
        
        print("\n✅ Création de la matrice des features terminée avec succès")
        print(f"   • {matrix.shape[1]} features")
        print(f"   • {matrix.shape[0]} échantillons")
        if not identifications.empty:
            print(f"   • {len(identifications)} identifications")
            n_with_ms2 = sum(len(ms2) > 0 for ms2 in identifications['peaks_mz_ms2'])
            print(f"   • {n_with_ms2} spectres MS2")
            print(f"   • Résultats sauvegardés dans {output_dir}")
            
            # Distribution des niveaux de confiance
            confidence_counts = identifications['confidence_level'].value_counts().sort_index()
            print("\n📊 Distribution des niveaux de confiance:")
            for level, count in confidence_counts.items():
                print(f"   • Niveau {level}: {count} identifications ({count/len(identifications)*100:.1f}%)")
        
    except Exception as e:
        print(f"\n❌ Erreur lors de la création de la matrice: {str(e)}")
        raise