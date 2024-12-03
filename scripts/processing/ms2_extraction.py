#scripts/processing/ms2_extraction.py
#-*- coding:utf-8 -*-


# -*- coding:utf-8 -*-

import logging
import pandas as pd
from pathlib import Path

# Initialiser le logger
logger = logging.getLogger(__name__)

def extract_ms2_for_matches(matches_df: pd.DataFrame, raw_parquet_path: str, output_dir: str, silent: bool = True) -> pd.DataFrame:
    """
    Extrait les spectres MS2 pour chaque correspondance dans un DataFrame de correspondances.

    Args:
        matches_df (pd.DataFrame): DataFrame contenant les correspondances pour lesquelles extraire les spectres MS2.
        raw_parquet_path (str): Chemin vers le fichier brut des données MS en format Parquet.
        output_dir (str): Répertoire où sauvegarder le fichier all_matches.parquet mis à jour.
        silent (bool): Si True, supprime les messages de progression.

    Returns:
        pd.DataFrame: DataFrame des correspondances mis à jour avec les colonnes peaks_mz_ms2 et peaks_intensities_ms2.

    Raises:
        Exception: Si une erreur survient pendant l'extraction ou la mise à jour des données.
    """
    try:
        # Charge les données brutes MS depuis le fichier Parquet
        raw_data = pd.read_parquet(path=raw_parquet_path)
        raw_data['mslevel'] = raw_data['mslevel'].astype(int)
        ms2_data = raw_data[raw_data['mslevel'] == 2]

        # Initialisation des listes pour stocker les spectres MS2 extraits
        peaks_mz_ms2_list = []
        peaks_intensities_ms2_list = []
        n_with_spectra = 0
        total_matches = len(matches_df)

        # Parcourt chaque correspondance dans le DataFrame
        for _, match in matches_df.iterrows():
            # Filtre les spectres MS2 correspondant aux critères RT, DT
            match_ms2 = ms2_data[
                (ms2_data['rt'] >= match['peak_rt'] - 0.00422) &
                (ms2_data['rt'] <= match['peak_rt'] + 0.00422) &
                (ms2_data['dt'] >= match['peak_dt'] - 0.22) &
                (ms2_data['dt'] <= match['peak_dt'] + 0.22)
            ]

            if len(match_ms2) > 0:
                # Regroupe les spectres MS2 en sommant les intensités par m/z arrondi
                match_ms2['mz_rounded'] = match_ms2['mz'].round(3)
                spectrum = match_ms2.groupby('mz_rounded')['intensity'].sum().reset_index()

                # Normalise les intensités
                max_intensity = spectrum['intensity'].max()
                if max_intensity > 0:
                    spectrum['intensity_normalized'] = (spectrum['intensity'] / max_intensity * 999).round(0).astype(int)
                    # Sélectionne les 10 pics les plus intenses
                    spectrum = spectrum.nlargest(10, 'intensity')
                    peaks_mz_ms2_list.append(spectrum['mz_rounded'].tolist())
                    peaks_intensities_ms2_list.append(spectrum['intensity_normalized'].tolist())
                    n_with_spectra += 1
                else:
                    peaks_mz_ms2_list.append([])
                    peaks_intensities_ms2_list.append([])
            else:
                peaks_mz_ms2_list.append([])
                peaks_intensities_ms2_list.append([])

        # Ajoute les colonnes MS2 au DataFrame original des correspondances
        matches_df['peaks_mz_ms2'] = peaks_mz_ms2_list
        matches_df['peaks_intensities_ms2'] = peaks_intensities_ms2_list

        # Crée le répertoire de sortie si nécessaire
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Sauvegarde le DataFrame mis à jour dans un fichier Parquet
        output_file = output_dir / 'all_matches.parquet'
        matches_df.to_parquet(output_file)

        if not silent:
            print("\n   ℹ️ Résultats de l'extraction MS2:")
            print(f"      - {n_with_spectra}/{total_matches} matches ont des spectres MS2 ({n_with_spectra/total_matches*100:.1f}%)")
            print(f"   ✓ Fichier all_matches.parquet mis à jour avec les spectres MS2")

        return matches_df

    except Exception as e:
        logger.error(f"Erreur lors de l'extraction MS2 : {str(e)}")
        if not silent:
            print(f"   ✗ Erreur : {str(e)}")
        raise