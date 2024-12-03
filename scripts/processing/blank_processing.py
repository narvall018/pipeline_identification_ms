# blank processing

from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from ..utils.io_handlers import read_parquet_data, save_peaks
from ..processing.peak_detection import prepare_data, detect_peaks, cluster_peaks


def process_blank_file(
    file_path: Union[str, Path],
    data_type: str = 'blanks',
    total_files: int = 1,
    current_file: int = 1
) -> Optional[pd.DataFrame]:
    """
    Traite un fichier blank pour détecter les pics (sans CCS).
    
    Args:
        file_path: Chemin vers le fichier blank
        data_type: Type de données (default: 'blanks')
        total_files: Nombre total de fichiers
        current_file: Index du fichier courant
        
    Returns:
        DataFrame contenant les pics du blank ou None si erreur
    """
    try:
        sample_name = Path(file_path).stem
        
        print(f"\n{'=' * 80}")
        print(f"TRAITEMENT DU BLANK {sample_name} ({current_file}/{total_files})")
        print(f"{'=' * 80}")

        # Lecture des données
        print("\n📊 Lecture des données...")
        data, metadata = read_parquet_data(file_path)
        print(f"   ✓ Données chargées : {len(data)} lignes")

        # Préparation MS1
        print("\n🔍 Préparation des données MS1...")
        processed_data = prepare_data(data)
        if processed_data is None or processed_data.empty:
            print("   ✗ Aucune donnée MS1 valide")
            return None
        print(f"   ✓ Données préparées : {len(processed_data)} lignes")

        # Détection des pics
        print("\n🎯 Détection des pics...")
        peaks = detect_peaks(processed_data)
        if peaks.empty:
            print("   ✗ Aucun pic détecté")
            return None
        print(f"   ✓ Pics détectés : {len(peaks)}")
        save_peaks(peaks, sample_name, "peaks", data_type, metadata)

        # Clustering des pics
        print("\n🔄 Clustering des pics...")
        clustered_peaks = cluster_peaks(peaks)
        if clustered_peaks.empty:
            print("   ✗ Pas de pics après clustering")
            return None
        print(f"   ✓ Pics après clustering : {len(clustered_peaks)}")
        save_peaks(clustered_peaks, sample_name, "clustered_peaks", data_type, metadata)
        
        return clustered_peaks

    except Exception as e:
        print(f"\n❌ Erreur lors du traitement de {sample_name}")
        raise

def process_blank_with_replicates(blank_name: str, 
                                replicate_files: List[Path],
                                output_dir: Path) -> pd.DataFrame:
    """
    Traite un blank avec ses réplicats.
    
    Args:
        blank_name: Nom de base du blank
        replicate_files: Liste des fichiers réplicats
        output_dir: Dossier de sortie
        
    Returns:
        DataFrame des pics communs aux réplicats
    """
    try:
        print(f"\n{'='*80}")
        print(f"Traitement du blank {blank_name}")
        print(f"{'='*80}")
        
        all_peaks = {}
        initial_peak_counts = {}
        
        # Traiter chaque réplicat
        for rep_file in replicate_files:
            peaks = process_blank_file(rep_file)
            if peaks is not None:
                all_peaks[rep_file.stem] = peaks
                initial_peak_counts[rep_file.stem] = len(peaks)
        
        if not all_peaks:
            return pd.DataFrame()
            
        # Si un seul réplicat
        if len(all_peaks) == 1:
            return list(all_peaks.values())[0]
            
        # Combiner les réplicats
        min_required = 2 if len(replicate_files) == 3 else len(replicate_files)
        print(f"   ℹ️ Critère: {min_required}/{len(replicate_files)} réplicats requis")
        
        return cluster_blank_replicates(all_peaks, min_required)
        
    except Exception as e:
        print(f"❌ Erreur lors du traitement du blank {blank_name}: {str(e)}")
        return pd.DataFrame()

def cluster_blank_replicates(peaks_dict: Dict[str, pd.DataFrame],  ## DBSCAN ## 
                           min_required: int) -> pd.DataFrame:
    """
    Cluster les pics entre réplicats de blanks.
    """
    # Combiner tous les pics
    all_peaks = []
    for rep_name, peaks in peaks_dict.items():
        peaks_copy = peaks.copy()
        peaks_copy['replicate'] = rep_name
        all_peaks.append(peaks_copy)
    
    combined_peaks = pd.concat(all_peaks, ignore_index=True)
    
    if len(combined_peaks) == 0:
        return pd.DataFrame()
    
    # Préparation pour DBSCAN
    X = combined_peaks[['mz', 'drift_time', 'retention_time']].values
    
    # Tolérances identiques aux échantillons
    mz_tolerance = np.median(X[:, 0]) * 1e-4  # 0.1 ppm
    dt_tolerance = np.median(X[:, 1]) * 0.10   # 10%
    rt_tolerance = 0.20                        # 0.2 min
    
    # Normalisation
    X_scaled = np.zeros_like(X)
    X_scaled[:, 0] = X[:, 0] / mz_tolerance
    X_scaled[:, 1] = X[:, 1] / dt_tolerance
    X_scaled[:, 2] = X[:, 2] / rt_tolerance
    
    # Clustering
    clusters = DBSCAN(eps=1.0, min_samples=min_required).fit_predict(X_scaled)
    combined_peaks['cluster'] = clusters
    
    # Traitement des clusters
    result = []
    for cluster_id in sorted(set(clusters)):
        if cluster_id == -1:
            continue
            
        cluster_data = combined_peaks[combined_peaks['cluster'] == cluster_id]
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


def subtract_blank_peaks(sample_peaks: pd.DataFrame, 
                        blank_peaks: pd.DataFrame) -> pd.DataFrame:
    """
    Soustrait les pics du blank des pics de l'échantillon.
    
    Args:
        sample_peaks: DataFrame des pics de l'échantillon
        blank_peaks: DataFrame des pics du blank
        
    Returns:
        DataFrame des pics de l'échantillon sans ceux du blank
    """
    if blank_peaks.empty or sample_peaks.empty:
        return sample_peaks
        
    print("\n🧹 Soustraction des pics du blank...")
    initial_peaks = len(sample_peaks)
    
    # Préparer les données pour DBSCAN
    combined = pd.concat([
        sample_peaks.assign(is_sample=True),
        blank_peaks.assign(is_sample=False)
    ], ignore_index=True)
    
    X = combined[['mz', 'drift_time', 'retention_time']].values
    
    # Mêmes tolérances que pour le clustering
    mz_tolerance = np.median(X[:, 0]) * 1e-4
    dt_tolerance = np.median(X[:, 1]) * 0.10
    rt_tolerance = 0.20
    
    X_scaled = np.zeros_like(X)
    X_scaled[:, 0] = X[:, 0] / mz_tolerance
    X_scaled[:, 1] = X[:, 1] / dt_tolerance
    X_scaled[:, 2] = X[:, 2] / rt_tolerance
    
    # Clustering
    clusters = DBSCAN(eps=1.0, min_samples=2).fit_predict(X_scaled)
    combined['cluster'] = clusters
    
    # Identifier les clusters contenant des pics du blank
    blank_clusters = set(combined[
        (~combined['is_sample']) & (combined['cluster'] != -1)
    ]['cluster'])
    
    # Filtrer les pics
    clean_peaks = combined[
        combined['is_sample'] & 
        (~combined['cluster'].isin(blank_clusters) | (combined['cluster'] == -1))
    ].copy()
    
    # Nettoyage
    clean_peaks = clean_peaks.drop(['is_sample', 'cluster'], axis=1)
    
    peaks_removed = initial_peaks - len(clean_peaks)
    print(f"   ✓ {peaks_removed} pics retirés ({peaks_removed/initial_peaks*100:.1f}%)")
    
    return clean_peaks
