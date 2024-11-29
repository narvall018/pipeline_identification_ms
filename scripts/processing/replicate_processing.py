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
    """Cluster les pics entre réplicats et calcule les valeurs représentatives"""
    # Si un seul réplicat, retourner directement ses pics
    if len(peaks_dict) == 1:
        return list(peaks_dict.values())[0]
    
    # Combiner tous les réplicats
    all_peaks = []
    for rep_name, peaks in peaks_dict.items():
        peaks_copy = peaks.copy()
        peaks_copy['replicate'] = rep_name
        all_peaks.append(peaks_copy)
    
    combined_peaks = pd.concat(all_peaks, ignore_index=True)
    
    if len(combined_peaks) == 0:
        return pd.DataFrame()
    
    # Critères pour le clustering
    total_replicates = len(peaks_dict)
    min_required = 2 if total_replicates == 3 else total_replicates  # 2/3 ou 2/2
    
    # Préparation pour DBSCAN
    X = combined_peaks[['mz', 'drift_time', 'retention_time']].values
    
    # Tolérances
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
        
        # Vérification des critères 2/2 ou 2/3
        if ((total_replicates == 2 and n_replicates == 2) or  # 2/2
            (total_replicates == 3 and n_replicates >= 2)):   # 2/3
            
            # NOUVEAU: Calcul des valeurs représentatives
            representative = {
                'mz': cluster_data['mz'].mean(),              # Moyenne mz
                'drift_time': cluster_data['drift_time'].mean(),  # Moyenne drift time
                'retention_time': cluster_data['retention_time'].mean(),  # Moyenne RT
                'intensity': cluster_data['intensity'].max(),  # Maximum intensity
                'CCS': cluster_data['CCS'].mean() if 'CCS' in cluster_data.columns else None,  # Moyenne CCS si présent
                'n_replicates': n_replicates
            }
            
            # Ajouter les autres colonnes si présentes
            for col in cluster_data.columns:
                if col not in ['mz', 'drift_time', 'retention_time', 'intensity', 'CCS', 'cluster', 'replicate']:
                    representative[col] = cluster_data[col].iloc[0]
            
            result.append(representative)
    
    result_df = pd.DataFrame(result) if result else pd.DataFrame()
    
    if not result_df.empty:
        result_df = result_df.sort_values('intensity', ascending=False)
    
    return result_df

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