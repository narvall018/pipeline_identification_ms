import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.cluster import DBSCAN
from .peak_detection import prepare_data, detect_peaks, cluster_peaks

def process_replicates(replicate_files: List[Path], calibrator) -> Tuple[Dict[str, pd.DataFrame], Dict[str, int]]:
    """
    Traite les réplicats d'un échantillon
    
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
            
            # Clustering et CCS
            clustered_peaks = cluster_peaks(peaks)
            peaks_with_ccs = calibrator.calculate_ccs(clustered_peaks)
            all_peaks[rep_file.stem] = peaks_with_ccs
            
            print(f"   ✓ {rep_file.stem}:")
            print(f"      - Pics initiaux: {initial_peak_counts[rep_file.stem]}")
            print(f"      - Pics après clustering: {len(peaks_with_ccs)}")
            
        except Exception as e:
            print(f"   ✗ Erreur avec {rep_file.stem}: {str(e)}")
    
    return all_peaks, initial_peak_counts

def cluster_replicates(peaks_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Cluster les pics entre réplicats"""
    # Si un seul réplicat, retourner directement ses pics
    if len(peaks_dict) == 1:
        return list(peaks_dict.values())[0]
    
    # Sinon, procéder au clustering entre réplicats
    all_peaks = []
    for rep_name, peaks in peaks_dict.items():
        peaks_copy = peaks.copy()
        peaks_copy['replicate'] = rep_name
        all_peaks.append(peaks_copy)
    
    combined_peaks = pd.concat(all_peaks, ignore_index=True)
    
    if len(combined_peaks) == 0:
        return pd.DataFrame()
    
    # Critères pour plusieurs réplicats
    total_replicates = len(peaks_dict)
    if total_replicates == 2:
        min_required = 2  # 2/2
        print(f"   ℹ️ Critère: {min_required}/{total_replicates} réplicats requis")
    elif total_replicates == 3:
        min_required = 2  # 2/3
        print(f"   ℹ️ Critère: {min_required}/{total_replicates} réplicats requis")
    else:
        return pd.DataFrame()
    
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
            max_intensity_idx = cluster_data['intensity'].idxmax()
            representative = cluster_data.loc[max_intensity_idx].copy()
            representative['n_replicates'] = n_replicates
            result.append(representative)
    
    result_df = pd.DataFrame(result) if result else pd.DataFrame()
    
    if not result_df.empty:
        result_df = result_df.sort_values('intensity', ascending=False)
    
    return result_df

def process_sample_with_replicates(sample_name: str, 
                                 replicate_files: List[Path],
                                 calibrator,
                                 output_dir: Path) -> pd.DataFrame:
    """Process complet pour un échantillon avec réplicats"""
    try:
        print(f"\n{'='*80}")
        print(f"Traitement de {sample_name}")
        print(f"{'='*80}")
        
        print(f"\n🔍 Traitement des réplicats ({len(replicate_files)} fichiers)...")
        
        # Traitement des réplicats
        peaks_data = process_replicates(replicate_files, calibrator)
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
        
        # Calcul CCS (déjà fait dans process_replicates mais affichage du message)
        print("\n🔵 Calibration CCS...")
        print(f"   ✓ CCS calculées pour {len(final_peaks)} pics")
        print(f"   ✓ Plage de CCS: {final_peaks['CCS'].min():.2f} - {final_peaks['CCS'].max():.2f} Å²")
        print(f"   ✓ CCS moyenne: {final_peaks['CCS'].mean():.2f} Å²")
        
        # Sauvegarde des pics
        output_dir = output_dir / sample_name / "ms1"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "common_peaks.parquet"
        final_peaks.to_parquet(output_file)
        print(f"   ✓ Résultats sauvegardés dans {output_file}")

        print("\n🔍 Identification des composés...")
        
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
        print(f"   - Pics avec CCS: {len(final_peaks)}")
        print(f"{'='*80}")
        
        return final_peaks
        
    except Exception as e:
        print(f"❌ Erreur lors du traitement de {sample_name}: {str(e)}")
        return pd.DataFrame()