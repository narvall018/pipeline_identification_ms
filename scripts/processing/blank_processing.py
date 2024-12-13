#scripts/processing/ccs_calibration.py
# -*- coding:utf-8 -*-

from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from ..utils.io_handlers import read_parquet_data, save_peaks
from ..processing.peak_detection import prepare_data, detect_peaks, cluster_peaks

def process_blank_file(
    file_path: Union[str, Path],
    data_type: str = 'blanks'
) -> Optional[pd.DataFrame]:
    data, metadata = read_parquet_data(file_path)
    processed_data = prepare_data(data)
    if processed_data is None or processed_data.empty:
        return None

    peaks = detect_peaks(processed_data)
    if peaks.empty:
        return None
    sample_name = Path(file_path).stem
    save_peaks(peaks, sample_name, "peaks", data_type, metadata)

    clustered_peaks = cluster_peaks(peaks)
    if clustered_peaks.empty:
        return None
    save_peaks(clustered_peaks, sample_name, "clustered_peaks", data_type, metadata)
    
    return clustered_peaks

def process_blank_with_replicates(
    blank_name: str, 
    replicate_files: List[Path],
    output_dir: Path
) -> pd.DataFrame:
    print(f"\n{'='*80}")
    print(f"TRAITEMENT DU BLANK {blank_name}")
    print(f"{'='*80}")

    all_peaks = {}
    for rep_file in replicate_files:
        # Renommer le fichier pour enlever "_replicate_"
        rep_name = rep_file.stem.replace('_replicate_', '_')
        
        peaks = process_blank_file(rep_file)
        if peaks is not None:
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
    combined_peaks = cluster_blank_replicates(all_peaks, min_required)
    print(f"\n   Pics finaux après convergence : {len(combined_peaks)}")

    return combined_peaks


def cluster_blank_replicates(peaks_dict: Dict[str, pd.DataFrame], min_required: int) -> pd.DataFrame:
    all_peaks = pd.concat(
        [peaks.assign(replicate=name) for name, peaks in peaks_dict.items()],
        ignore_index=True
    )
    
    if len(all_peaks) == 0:
        return pd.DataFrame()
    
    X = all_peaks[['mz', 'drift_time', 'retention_time']].to_numpy()
    median_mz = np.median(X[:, 0])
    mz_tolerance = median_mz * 10e-6
    X_scaled = np.column_stack([
        X[:, 0] / mz_tolerance,
        X[:, 1] / 1.0,
        X[:, 2] / 0.1
    ])
    
    clusters = DBSCAN(
        eps=0.6,
        min_samples=min_required,
        algorithm='ball_tree',
        n_jobs=-1
    ).fit_predict(X_scaled)
    
    all_peaks['cluster'] = clusters
    valid_clusters = all_peaks[clusters != -1].groupby('cluster')
    
    result = []
    for _, cluster_data in valid_clusters:
        n_replicates = cluster_data['replicate'].nunique()
        if n_replicates >= min_required:
            max_intensity_idx = cluster_data['intensity'].idxmax()
            representative = cluster_data.loc[max_intensity_idx].copy()
            representative['n_replicates'] = n_replicates
            result.append(representative)
    
    result_df = pd.DataFrame(result) if result else pd.DataFrame()
    if not result_df.empty:
        result_df = result_df.sort_values('intensity', ascending=False)
    
    return result_df

def subtract_blank_peaks(sample_peaks: pd.DataFrame, blank_peaks: pd.DataFrame) -> pd.DataFrame:
    if blank_peaks.empty or sample_peaks.empty:
        return sample_peaks

    combined = pd.concat(
        [sample_peaks.assign(is_sample=True), blank_peaks.assign(is_sample=False)],
        ignore_index=True
    )
    
    if combined.empty:
        return sample_peaks

    X = combined[['mz', 'drift_time', 'retention_time']].values
    median_mz = np.median(X[:, 0])
    mz_tolerance = median_mz * 10e-6

    X_scaled = np.column_stack([
        X[:, 0] / mz_tolerance,
        X[:, 1] / 1.0,
        X[:, 2] / 0.1
    ])

    # Utilisation de ball_tree et parallélisation
    clusters = DBSCAN(
        eps=1.0,
        min_samples=2,
        algorithm='ball_tree',
        n_jobs=-1
    ).fit_predict(X_scaled)
    
    combined['cluster'] = clusters

    blank_clusters = set(
        combined[(~combined['is_sample']) & (combined['cluster'] != -1)]['cluster']
    )

    clean_peaks = combined[
        combined['is_sample'] & 
        (~combined['cluster'].isin(blank_clusters) | (combined['cluster'] == -1))
    ].copy()

    clean_peaks = clean_peaks.drop(['is_sample', 'cluster'], axis=1)
    return clean_peaks



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


# def cluster_blank_replicates(peaks_dict: Dict[str, pd.DataFrame], 
#                            min_required: int) -> pd.DataFrame:
#     """Cluster les pics entre réplicats de blanks."""
#     # Combiner tous les pics
#     all_peaks = []
#     for rep_name, peaks in peaks_dict.items():
#         peaks_copy = peaks.copy()
#         peaks_copy['replicate'] = rep_name
#         all_peaks.append(peaks_copy)
    
#     combined_peaks = pd.concat(all_peaks, ignore_index=True)
    
#     if len(combined_peaks) == 0:
#         return pd.DataFrame()

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
        
#         if n_replicates >= min_required:
#             representative = cluster_data.loc[cluster_data['intensity'].idxmax()].copy()
#             representative = representative.drop(['cluster', 'distance', 'replicate'])
#             representative['n_replicates'] = n_replicates
#             result.append(representative)
    
#     result_df = pd.DataFrame(result) if result else pd.DataFrame()
    
#     if not result_df.empty:
#         result_df = result_df.sort_values('intensity', ascending=False)
    
#     return result_df



