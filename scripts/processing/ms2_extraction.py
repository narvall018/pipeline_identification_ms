#scripts/processing/ms2_extraction.py
#-*- coding:utf-8 -*-

import logging
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from ..config.config import Config

class MS2Extractor:
    """Classe responsable de l'extraction des spectres MS2."""

    def __init__(self):
        """Initialise l'extracteur MS2 avec la configuration."""
        self.config = Config.MS2_EXTRACTION
        self.logger = logging.getLogger(__name__)

    def extract_ms2_spectrum(
        self,
        ms2_data: pd.DataFrame,
        rt: float,
        dt: float
    ) -> Tuple[List[float], List[int]]:
        """
        Extrait un spectre MS2 pour une paire RT/DT donnée.
        
        Args:
            ms2_data: Données MS2 brutes
            rt: Temps de rétention cible
            dt: Temps de dérive cible
            
        Returns:
            Tuple[List[float], List[int]]: m/z et intensités normalisées
        """
        try:
            # Définir la fenêtre de recherche
            rt_min = rt - self.config.rt_tolerance
            rt_max = rt + self.config.rt_tolerance
            dt_min = dt - self.config.dt_tolerance
            dt_max = dt + self.config.dt_tolerance

            # Filtrer les données dans la fenêtre
            sub_data = ms2_data[
                (ms2_data['rt'] >= rt_min) &
                (ms2_data['rt'] <= rt_max) &
                (ms2_data['dt'] >= dt_min) &
                (ms2_data['dt'] <= dt_max)
            ]

            if len(sub_data) == 0:
                return [], []

            # Arrondir et regrouper les m/z
            sub_data['mz_rounded'] = sub_data['mz'].round(self.config.mz_round_decimals)
            spectrum = sub_data.groupby('mz_rounded')['intensity'].sum().reset_index()

            # Normaliser les intensités
            max_intensity = spectrum['intensity'].max()
            if max_intensity > 0:
                spectrum['intensity_normalized'] = (
                    spectrum['intensity'] / max_intensity * 
                    self.config.intensity_scale
                ).round(0).astype(int)
                
                # Prendre les N pics les plus intenses
                spectrum = spectrum.nlargest(self.config.max_peaks, 'intensity')
                return (
                    spectrum['mz_rounded'].tolist(),
                    spectrum['intensity_normalized'].tolist()
                )

            return [], []

        except Exception as e:
            self.logger.error(f"Erreur lors de l'extraction du spectre MS2: {str(e)}")
            return [], []

    def extract_ms2_for_matches(
        self,
        matches_df: pd.DataFrame,
        raw_parquet_path: str,
        output_dir: str,
        silent: bool = True
    ) -> pd.DataFrame:
        """
        Extrait les spectres MS2 pour chaque correspondance dans un DataFrame.
        
        Args:
            matches_df: DataFrame des correspondances
            raw_parquet_path: Chemin vers le fichier brut
            output_dir: Répertoire de sortie
            silent: Si True, supprime les messages de progression
            
        Returns:
            pd.DataFrame: DataFrame mis à jour avec les spectres MS2
        """
        try:
            # Charger les données MS2
            raw_data = pd.read_parquet(raw_parquet_path)
            raw_data['mslevel'] = raw_data['mslevel'].astype(int)
            ms2_data = raw_data[raw_data['mslevel'] == 2]

            # Initialisation des listes pour stocker les spectres
            peaks_mz_ms2_list = []
            peaks_intensities_ms2_list = []
            n_with_spectra = 0
            total_matches = len(matches_df)

            # Parcourir chaque correspondance
            for idx, match in matches_df.iterrows():
                mzs, intensities = self.extract_ms2_spectrum(
                    ms2_data,
                    match['peak_rt'],
                    match['peak_dt']
                )
                
                if mzs:  # Si un spectre a été trouvé
                    peaks_mz_ms2_list.append(mzs)
                    peaks_intensities_ms2_list.append(intensities)
                    n_with_spectra += 1
                else:
                    peaks_mz_ms2_list.append([])
                    peaks_intensities_ms2_list.append([])

            # Mise à jour du DataFrame
            matches_df['peaks_mz_ms2'] = peaks_mz_ms2_list
            matches_df['peaks_intensities_ms2'] = peaks_intensities_ms2_list

            # Sauvegarde des résultats
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / 'all_matches.parquet'
            matches_df.to_parquet(output_file)

            if not silent:
                print("\n   ℹ️ Résultats de l'extraction MS2:")
                print(f"      - {n_with_spectra}/{total_matches} matches ont des spectres MS2"
                      f" ({n_with_spectra/total_matches*100:.1f}%)")
                print(f"   ✓ Fichier all_matches.parquet mis à jour avec les spectres MS2")

            return matches_df

        except Exception as e:
            error_msg = f"Erreur lors de l'extraction MS2 : {str(e)}"
            self.logger.error(error_msg)
            if not silent:
                print(f"   ✗ Erreur : {error_msg}")
            raise

    def batch_extract_ms2(
        self,
        matches_df: pd.DataFrame,
        raw_data: pd.DataFrame,
        progress_callback: Optional[callable] = None
    ) -> pd.DataFrame:
        """
        Extrait les spectres MS2 par lots pour optimiser les performances.
        
        Args:
            matches_df: DataFrame des correspondances
            raw_data: Données MS brutes
            progress_callback: Fonction optionnelle pour suivre la progression
            
        Returns:
            pd.DataFrame: DataFrame avec les spectres MS2 extraits
        """
        try:
            # Préparation des données MS2
            ms2_data = raw_data[raw_data['mslevel'] == 2].copy()
            if ms2_data.empty:
                self.logger.warning("Aucune donnée MS2 trouvée")
                return matches_df

            results = []
            total = len(matches_df)
            
            for i, match in enumerate(matches_df.itertuples()):
                mzs, intensities = self.extract_ms2_spectrum(
                    ms2_data,
                    match.peak_rt,
                    match.peak_dt
                )
                
                results.append({
                    'index': match.Index,
                    'peaks_mz_ms2': mzs,
                    'peaks_intensities_ms2': intensities
                })
                
                if progress_callback and i % 100 == 0:
                    progress_callback(i / total)

            # Mise à jour du DataFrame original
            result_df = pd.DataFrame(results).set_index('index')
            matches_df = matches_df.join(result_df)

            return matches_df

        except Exception as e:
            self.logger.error(f"Erreur lors de l'extraction MS2 par lots : {str(e)}")
            raise