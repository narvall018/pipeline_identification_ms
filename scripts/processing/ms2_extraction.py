# scripts/processing/ms2_extraction.py
import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def extract_ms2_for_matches(matches_df, raw_parquet_path, output_dir):
    """
    Extrait les spectres MS2 pour chaque match et met à jour all_matches.parquet
    """
    try:
        print("\n🔍 Lecture du fichier brut pour MS2...")
        raw_data = pd.read_parquet(raw_parquet_path)
        
        # Convertir mslevel en entier si nécessaire
        raw_data['mslevel'] = raw_data['mslevel'].astype(int)
        
        # Afficher la distribution des niveaux MS
        ms_counts = raw_data['mslevel'].value_counts().sort_index()
        print(f"   ℹ️ Distribution des niveaux MS: \n{ms_counts}")
        
        # Filtrer MS2
        ms2_data = raw_data[raw_data['mslevel'] == 2]
        print(f"   ✓ Nombre de spectres MS2 disponibles: {len(ms2_data)}")
        
        # Initialiser les listes pour les colonnes MS2
        peaks_mz_ms2_list = []
        peaks_intensities_ms2_list = []
        
        print("\n🎯 Extraction des spectres MS2...")
        matches_with_ms2 = 0
        total_matches = len(matches_df)
        
        for idx, match in matches_df.iterrows():
            # Filtrer par RT, DT et m/z précurseur
            match_ms2 = ms2_data[
                (ms2_data['rt'] >= match['peak_rt'] - 0.00422) &
                (ms2_data['rt'] <= match['peak_rt'] + 0.00422) &
                (ms2_data['dt'] >= match['peak_dt'] - 0.22) &
                (ms2_data['dt'] <= match['peak_dt'] + 0.22)
            ]
            
            if len(match_ms2) > 0:
                matches_with_ms2 += 1
                # Traitement du spectre
                match_ms2['mz_rounded'] = match_ms2['mz'].round(3)
                spectrum = match_ms2.groupby('mz_rounded')['intensity'].sum().reset_index()
                
                # Normalisation
                max_intensity = spectrum['intensity'].max()
                if max_intensity > 0:
                    spectrum['intensity_normalized'] = (spectrum['intensity'] / max_intensity * 999).round(0).astype(int)
                    
                    # Top 10 pics
                    spectrum = spectrum.nlargest(10, 'intensity')
                    
                    # Sauvegarder les listes pour ce match
                    peaks_mz_ms2_list.append(spectrum['mz_rounded'].tolist())
                    peaks_intensities_ms2_list.append(spectrum['intensity_normalized'].tolist())
                else:
                    peaks_mz_ms2_list.append([])
                    peaks_intensities_ms2_list.append([])
            else:
                peaks_mz_ms2_list.append([])
                peaks_intensities_ms2_list.append([])
                
        
        # Ajouter les nouvelles colonnes au DataFrame original
        matches_df['peaks_mz_ms2'] = peaks_mz_ms2_list
        matches_df['peaks_intensities_ms2'] = peaks_intensities_ms2_list
        
        # Sauvegarder en écrasant le fichier all_matches.parquet existant
        output_file = output_dir / 'all_matches.parquet'
        matches_df.to_parquet(output_file)
        
        # Afficher les statistiques
        n_with_spectra = sum(len(mz_list) > 0 for mz_list in peaks_mz_ms2_list)
        print(f"\n   ℹ️ Résultats de l'extraction MS2:")
        print(f"      - {n_with_spectra}/{total_matches} matches ont des spectres MS2 ({n_with_spectra/total_matches*100:.1f}%)")
        print(f"   ✓ Fichier all_matches.parquet mis à jour avec les spectres MS2")
        
        return matches_df
            
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction MS2: {str(e)}")
        print(f"   ✗ Erreur: {str(e)}")
        raise
