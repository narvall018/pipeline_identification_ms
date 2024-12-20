#scripts/processing/replicate_processing.py
#-*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.cluster import DBSCAN
from .peak_detection import prepare_data, detect_peaks, cluster_peaks

def process_replicates(replicate_files: List[Path]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, int]]:
    """
    Traite les réplicats d'un échantillon sans la calibration CCS
    
    Returns:
        Tuple[Dict[str, pd.DataFrame], Dict[str, int]]: (peaks_dict, initial_peaks)
    """
    all_peaks = {}
    initial_peak_counts = {}
    
    for rep_file in replicate_files:
        try:
            # Process each replicate
            data = pd.read_parquet(rep_file)
            processed_data = prepare_data(data)
            peaks = detect_peaks(processed_data)
            
            # Stocker le nombre de pics avant clustering
            initial_peak_counts[rep_file.stem] = len(peaks)
            
            # Clustering uniquement
            clustered_peaks = cluster_peaks(peaks)
            all_peaks[rep_file.stem] = clustered_peaks
            
            print(f"   ✓ {rep_file.stem}:")
            print(f"      - Pics initiaux: {initial_peak_counts[rep_file.stem]}")
            print(f"      - Pics après clustering: {len(clustered_peaks)}")
            
        except Exception as e:
            print(f"   ✗ Erreur avec {rep_file.stem}: {str(e)}")
    
    return all_peaks, initial_peak_counts


def cluster_replicates(peaks_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
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
    # Calcul des tolérances
    mz_tolerance = median_mz * 10e-6
    X_scaled = np.column_stack([
        X[:, 0] / mz_tolerance,
        X[:, 1] / 1.0,  # dt_tolerance
        X[:, 2] / 0.1   # rt_tolerance
    ])
    
    # Clustering optimisé
    clusters = DBSCAN(
        eps=0.6,
        min_samples=min_required,
        algorithm='ball_tree',
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


## BIJECTION ## 

# def quotient_compute(a: float, b: float) -> float:
#     """
#     Calcule un quotient relatif représentant l'écart proportionnel entre deux valeurs.
#     """
#     if b == 0 or a == 0:
#         raise ValueError("Une division par zéro n'est pas possible.")
#     return 1 - (a / b) if a < b else 1 - (b / a)

# def compute_distance(row: np.ndarray, candidate: np.ndarray, dims: list, bij_tables: list) -> float:
#     """
#     Calcule la distance euclidienne entre deux points dans des dimensions spécifiées.
#     """
#     if len(bij_tables) == 1:
#         return np.sqrt(sum(
#             (row[bij_tables[0][dim]] - candidate[bij_tables[0][dim]])**2
#             for dim in dims
#         ))
#     return np.sqrt(sum(
#         (row[bij_tables[0][dim]] - candidate[bij_tables[1][dim]])**2
#         for dim in dims
#     ))



# def cluster_replicates(peaks_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
#     """Cluster les pics entre réplicats et calcule les valeurs représentatives"""
#     if len(peaks_dict) == 1:
#         return list(peaks_dict.values())[0]
    
#     # Combiner tous les réplicats
#     all_peaks = []
#     for rep_name, peaks in peaks_dict.items():
#         peaks_copy = peaks.copy()
#         peaks_copy['replicate'] = rep_name
#         all_peaks.append(peaks_copy)
    
#     combined_peaks = pd.concat(all_peaks, ignore_index=True)
    
#     if len(combined_peaks) == 0:
#         return pd.DataFrame()
    
#     # Critères pour le clustering
#     total_replicates = len(peaks_dict)
#     min_required = 2 if total_replicates == 3 else total_replicates  # 2/3 ou 2/2
    
#     # Définit les tolérances pour le clustering
#     tolerances = {"mz": 1e-4, "retention_time": 0.10, "drift_time": 0.20}

#     # Initialise les colonnes nécessaires pour le clustering
#     combined_peaks["cluster"] = -1
#     combined_peaks["distance"] = np.inf
#     combined_peaks = combined_peaks.sort_values(by=["intensity"], ascending=False).reset_index(drop=True)

#     # Convertit en array numpy
#     peaks_array = combined_peaks.to_numpy()

#     # Créer les tables de bijection
#     tl_bijection = {dim: idx for idx, dim in enumerate(tolerances.keys())}
#     df_bijection = {dim: idx for idx, dim in enumerate(combined_peaks.columns)}

#     # Clustering
#     cluster_id = 0
#     for i, row in enumerate(peaks_array):
#         if row[df_bijection["cluster"]] == -1:
#             row[df_bijection["cluster"]] = cluster_id

#             for j, candidate in enumerate(peaks_array):
#                 if i != j:
#                     is_within_threshold = all(
#                         quotient_compute(
#                             row[df_bijection[dim]],
#                             candidate[df_bijection[dim]]
#                         ) <= tolerances[dim]
#                         for dim in tolerances.keys()
#                     )

#                     if is_within_threshold:
#                         distance = compute_distance(
#                             row=row,
#                             candidate=candidate,
#                             dims=list(tl_bijection.keys()),
#                             bij_tables=[df_bijection]
#                         )

#                         if distance < peaks_array[j, df_bijection["distance"]]:
#                             peaks_array[j, df_bijection["cluster"]] = cluster_id
#                             peaks_array[j, df_bijection["distance"]] = distance

#             cluster_id += 1

#     # Reconvertir en DataFrame
#     combined_peaks = pd.DataFrame(data=peaks_array, columns=combined_peaks.columns)
    
#     # Traitement des clusters
#     result = []
#     for cluster_id in sorted(set(combined_peaks['cluster'])):
#         if cluster_id == -1:
#             continue
            
#         cluster_data = combined_peaks[combined_peaks['cluster'] == cluster_id]
#         n_replicates = cluster_data['replicate'].nunique()
        
#         if ((total_replicates == 2 and n_replicates == 2) or  # 2/2
#             (total_replicates == 3 and n_replicates >= 2)):   # 2/3
            
#             representative = {
#                 'mz': cluster_data['mz'].mean(),
#                 'drift_time': cluster_data['drift_time'].mean(),
#                 'retention_time': cluster_data['retention_time'].mean(),
#                 'intensity': cluster_data['intensity'].max(),
#                 'CCS': cluster_data['CCS'].mean() if 'CCS' in cluster_data.columns else None,
#                 'n_replicates': n_replicates
#             }
            
#             for col in cluster_data.columns:
#                 if col not in ['mz', 'drift_time', 'retention_time', 'intensity', 'CCS', 'cluster', 'replicate', 'distance']:
#                     representative[col] = cluster_data[col].iloc[0]
            
#             result.append(representative)
    
#     result_df = pd.DataFrame(result) if result else pd.DataFrame()
    
#     if not result_df.empty:
#         result_df = result_df.sort_values('intensity', ascending=False)
    
#     return result_df

def process_sample_with_replicates(sample_name: str, 
                                 replicate_files: List[Path],
                                 output_dir: Path) -> pd.DataFrame:
    """Process des réplicats sans calibration CCS"""
    try:
        print(f"\n{'='*80}")
        print(f"Traitement de {sample_name}")
        print(f"{'='*80}")
        
        print(f"\n🔍 Traitement des réplicats ({len(replicate_files)} fichiers)...")
        
        # Traitement des réplicats
        peaks_data = process_replicates(replicate_files)
        peaks_dict, initial_peaks = peaks_data
        
        if not peaks_dict:
            print("   ✗ Aucun pic trouvé")
            return pd.DataFrame()
            
        # Clustering ou pics directs selon le nombre de réplicats
        if len(replicate_files) > 1:
            print(f"\n🔄 Clustering des pics entre réplicats...")
            final_peaks = cluster_replicates(peaks_dict)
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