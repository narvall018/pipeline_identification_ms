# -*- coding:utf-8 -*-

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.cluster import DBSCAN
from ..processing.identification import CompoundIdentifier

# Initialiser le logger
logger = logging.getLogger(__name__)

def align_features_across_samples(samples_dir: Path, identifier: CompoundIdentifier) -> Tuple[pd.DataFrame, pd.DataFrame]:
   """
   Aligne les features et ajoute l'identification.
   """
   print("\n🔄 Alignement des features entre échantillons...")
   
   # Collecter tous les pics
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
                   all_peaks.append(peaks)
                   sample_names.append(sample_dir.name)
   
   if not all_peaks:
       raise ValueError("Aucun pic trouvé dans les échantillons")
   
   df = pd.concat(all_peaks, ignore_index=True)
   print(f"   ✓ Total: {len(df)} pics à travers {len(sample_names)} échantillons")
   
   print("\n🎯 Clustering des features...")
   
   # Préparation pour DBSCAN
   X = df[['mz', 'drift_time', 'retention_time']].values
   
   # Calculer les tolérances
   mz_tolerance = np.median(X[:, 0]) * 1e-4
   dt_tolerance = np.median(X[:, 1]) * 0.10
   rt_tolerance = 0.20
   
   # Normalisation
   X_scaled = np.zeros_like(X)
   X_scaled[:, 0] = X[:, 0] / mz_tolerance
   X_scaled[:, 1] = X[:, 1] / dt_tolerance
   X_scaled[:, 2] = X[:, 2] / rt_tolerance
   
   # Clustering avec DBSCAN
   clusters = DBSCAN(eps=1.0, min_samples=1).fit_predict(X_scaled)
   df['cluster'] = clusters
   
   # Calculer les moyennes par cluster
   cluster_means = df.groupby('cluster').agg({
       'mz': 'mean',
       'retention_time': 'mean',
       'drift_time': 'mean',
       'CCS': 'mean',
       'intensity': 'max'  
   }).round(4)
   
   print(f"   ✓ {len(cluster_means)} features uniques détectées")
   
   # Transformer cluster_means en format attendu par identify_compounds
   peaks_for_identification = cluster_means.rename(columns={
       'retention_time': 'retention_time',
       'drift_time': 'drift_time',
       'intensity': 'intensity',
       'CCS': 'CCS'
   }).reset_index(drop=True)
   
   # Identification des clusters
   print("\n🔍 Identification des features...")
   matches_df = identifier.identify_compounds(peaks_for_identification, "temp")
   if matches_df is not None and not matches_df.empty:
       cluster_means['identification'] = None
       for idx, row in cluster_means.iterrows():
           matching_rows = matches_df[
               (abs(matches_df['peak_mz'] - row['mz']) <= mz_tolerance) &
               (abs(matches_df['peak_rt'] - row['retention_time']) <= rt_tolerance) &
               (abs(matches_df['peak_ccs'] - row['CCS']) <= row['CCS'] * 0.12)
           ]
           
           if not matching_rows.empty:
               cluster_means.loc[idx, 'identification'] = matching_rows.iloc[0]['match_name']
   
   print("\n📊 Création de la matrice d'intensités...")
   intensity_matrix = pd.DataFrame(index=sample_names)
   
   for cluster_id in sorted(df['cluster'].unique()):
       if cluster_id == -1:
           continue
       
       feature_means = cluster_means.loc[cluster_id]
       
       # Création du nom de la feature
       if pd.notna(feature_means.get('identification')):
           feature_name = f"{feature_means['identification']}_mz{feature_means['mz']:.4f}_rt{feature_means['retention_time']:.2f}_dt{feature_means['drift_time']:.2f}_ccs{feature_means['CCS']:.1f}"
       else:
           feature_name = f"mz{feature_means['mz']:.4f}_rt{feature_means['retention_time']:.2f}_dt{feature_means['drift_time']:.2f}_ccs{feature_means['CCS']:.1f}"
       
       cluster_data = df[df['cluster'] == cluster_id]
       sample_intensities = cluster_data.groupby('sample')['intensity'].max()
       
       intensity_matrix[feature_name] = pd.Series(sample_intensities)
   
   # Trier par taux de remplissage
   fill_rates = intensity_matrix.notna().mean()
   intensity_matrix = intensity_matrix[fill_rates.sort_values(ascending=False).index]
   
   # Mettre à jour cluster_means pour maintenir le même ordre
   cluster_means = cluster_means.loc[[i for i in sorted(df['cluster'].unique()) if i != -1]].reset_index(drop=True)
   
   # Compter les identifications réussies
   if 'identification' in cluster_means.columns:
       n_identified = cluster_means['identification'].notna().sum()
       print(f"   ✓ {n_identified} features identifiées sur {len(cluster_means)}")
   
   return intensity_matrix, cluster_means
def save_feature_matrix(
    matrix: pd.DataFrame,
    cluster_means: pd.DataFrame,
    output_dir: Path
) -> None:
    """
    Sauvegarde la matrice des features et les statistiques associées.
    """
    print("\n💾 Sauvegarde des résultats...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarder les fichiers
    matrix.to_parquet(output_dir / "feature_matrix.parquet")
    matrix.to_csv(output_dir / "feature_matrix.csv")
    cluster_means.to_csv(output_dir / "cluster_means.csv")
    
    # Statistiques
    stats = pd.DataFrame({
        'features_total': [matrix.shape[1]],
        'samples_total': [matrix.shape[0]],
        'features_per_sample_mean': [matrix.notna().sum(axis=1).mean()],
        'features_per_sample_std': [matrix.notna().sum(axis=1).std()],
        'samples_per_feature_mean': [matrix.notna().sum(axis=0).mean()],
        'samples_per_feature_std': [matrix.notna().sum(axis=0).std()]
    })
    
    stats.to_csv(output_dir / "feature_matrix_stats.csv")
    print(f"   ✓ Résultats sauvegardés dans {output_dir}")
    
def create_feature_matrix(input_dir: Path, output_dir: Path, identifier: CompoundIdentifier) -> None:
    """
    Fonction principale pour créer et sauvegarder la matrice des features.
    """
    try:
        matrix, cluster_means = align_features_across_samples(input_dir, identifier)
        save_feature_matrix(matrix, cluster_means, output_dir)
        
        print("\n✅ Création de la matrice des features terminée avec succès")
        print(f"   • {matrix.shape[1]} features")
        print(f"   • {matrix.shape[0]} échantillons")
        print(f"   • Taux de remplissage: {(matrix.notna().sum().sum() / (matrix.shape[0] * matrix.shape[1]) * 100):.1f}%")
        
    except Exception as e:
        print(f"\n❌ Erreur lors de la création de la matrice: {str(e)}")
        raise