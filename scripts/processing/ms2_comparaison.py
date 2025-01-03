#scripts/processing/ms2_comparaison.py
#-*- coding:utf-8 -*-

import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from scipy.spatial.distance import cosine
from ..config.config import Config
from ..utils.matching_utils import assign_confidence_level
from tqdm import tqdm

class MS2Comparator:
    """Classe pour comparer des spectres MS2 et calculer leurs similarités."""
    
    def __init__(self, tolerance_mz: float = 0.01):
        """
        Initialise le comparateur MS2.
        
        Args:
            tolerance_mz: Tolérance en m/z pour la comparaison des pics (en Da)
        """
        self.tolerance_mz = tolerance_mz
        self.logger = logging.getLogger(__name__)

    def normalize_spectrum(
        self,
        mz_list: List[float],
        intensity_list: List[float]
    ) -> Tuple[List[float], List[float]]:
        """
        Normalise les intensités d'un spectre par rapport au pic le plus intense.
        
        Args:
            mz_list: Liste des m/z
            intensity_list: Liste des intensités correspondantes
            
        Returns:
            Tuple[List[float], List[float]]: m/z et intensités normalisées
        """
        try:
            # Vérification des listes vides
            if not mz_list or not intensity_list:
                return [], []

            # Normalisation
            intensity_array = np.array(intensity_list)
            max_intensity = np.max(intensity_array)

            if max_intensity == 0:
                return [], []

            normalized_intensities = (intensity_array / max_intensity) * 1000

            return mz_list, normalized_intensities.tolist()

        except Exception as e:
            self.logger.error(f"Erreur dans la normalisation du spectre : {str(e)}")
            return [], []

    def align_spectra(
        self,
        exp_mz: List[float],
        exp_int: List[float],
        ref_mz: List[float],
        ref_int: List[float]
    ) -> Tuple[List[float], List[float]]:
        """
        Aligne deux spectres en fonction des m/z communs.
        
        Args:
            exp_mz: m/z expérimentaux
            exp_int: Intensités expérimentales
            ref_mz: m/z de référence
            ref_int: Intensités de référence
            
        Returns:
            Tuple[List[float], List[float]]: Intensités alignées
        """
        if not exp_mz or not ref_mz:
            return [], []

        aligned_exp_int = []
        aligned_ref_int = []

        for i, mz_exp in enumerate(exp_mz):
            matched = False
            for j, mz_ref in enumerate(ref_mz):
                if abs(mz_exp - mz_ref) <= self.tolerance_mz:
                    aligned_exp_int.append(exp_int[i])
                    aligned_ref_int.append(ref_int[j])
                    matched = True
                    break
            
            if not matched:
                aligned_exp_int.append(exp_int[i])
                aligned_ref_int.append(0)

        for j, mz_ref in enumerate(ref_mz):
            if all(abs(mz_exp - mz_ref) > self.tolerance_mz for mz_exp in exp_mz):
                aligned_exp_int.append(0)
                aligned_ref_int.append(ref_int[j])

        return aligned_exp_int, aligned_ref_int

    def calculate_similarity_score(
        self,
        exp_mz: List[float],
        exp_int: List[float],
        ref_mz: List[float],
        ref_int: List[float]
    ) -> float:
        """
        Calcule le score de similarité entre deux spectres.
        
        Args:
            exp_mz: m/z expérimentaux
            exp_int: Intensités expérimentales
            ref_mz: m/z de référence
            ref_int: Intensités de référence
            
        Returns:
            float: Score de similarité (0-1)
        """
        try:
            # Vérification et conversion des données d'entrée
            exp_mz = exp_mz if isinstance(exp_mz, list) else ([] if pd.isna(exp_mz).any() else exp_mz.tolist())
            exp_int = exp_int if isinstance(exp_int, list) else ([] if pd.isna(exp_int).any() else exp_int.tolist())
            ref_mz = ref_mz if isinstance(ref_mz, list) else ([] if pd.isna(ref_mz).any() else ref_mz.tolist())
            ref_int = ref_int if isinstance(ref_int, list) else ([] if pd.isna(ref_int).any() else ref_int.tolist())

            if not exp_mz or not ref_mz:
                return 0.0

            # Normalisation
            exp_mz, exp_int = self.normalize_spectrum(exp_mz, exp_int)
            ref_mz, ref_int = self.normalize_spectrum(ref_mz, ref_int)

            if not exp_int or not ref_int:
                return 0.0

            # Alignement
            aligned_exp, aligned_ref = self.align_spectra(exp_mz, exp_int, ref_mz, ref_int)

            if not aligned_exp or not aligned_ref:
                return 0.0

            # Calcul de similarité
            similarity = 1 - cosine(aligned_exp, aligned_ref)
            return max(0, similarity)

        except Exception as e:
            self.logger.error(f"Erreur dans le calcul du score de similarité : {str(e)}")
            return 0.0


def add_ms2_scores(matches_df: pd.DataFrame, identifier: object) -> None:
    """
    Ajoute les scores de similarité MS2 et recalcule les niveaux de confiance.
    
    Args:
        matches_df: DataFrame des correspondances
        identifier: Instance de CompoundIdentifier
    """
    try:
        print("\n🔬 Analyse des spectres MS2...")
        comparator = MS2Comparator(tolerance_mz=0.01)
        
        # Initialiser la colonne ms2_similarity_score avec 0
        matches_df['ms2_similarity_score'] = 0.0
        
        # Garder une copie des niveaux 1 existants
        level1_mask = matches_df['confidence_level'] == 1
        level1_indices = matches_df[level1_mask].index
        
        # Pré-filtrer les matches à analyser
        matches_to_analyze = matches_df[
            (~level1_mask) &
            (matches_df['has_ms2_db'] == 1) &
            matches_df['peaks_mz_ms2'].apply(lambda x: isinstance(x, list) and len(x) > 0)
        ]
        
        n_matches_with_ms2 = len(matches_to_analyze)
        print(f"   ✓ {n_matches_with_ms2}/{len(matches_df)} matches avec MS2 à analyser"
              " (spectres exp + DB)")
        
        # Créer un cache pour les spectres de référence
        ms2_ref_cache = {}
        
        # Traiter les matches sélectionnés
        for idx in tqdm(matches_to_analyze.index, desc="Calcul scores MS2"):
            row = matches_df.loc[idx]
            best_score = 0.0
            
            # Vérifier le cache
            cache_key = f"{row['match_name']}_{row['match_adduct']}"
            if cache_key not in ms2_ref_cache:
                ref_spectra = identifier.db[
                    (identifier.db['Name'] == row['match_name']) & 
                    (identifier.db['adduct'] == row['match_adduct'])
                ]
                ms2_ref_cache[cache_key] = ref_spectra
            else:
                ref_spectra = ms2_ref_cache[cache_key]

            # Comparer avec chaque spectre de référence
            for _, ref_row in ref_spectra.iterrows():
                if not (
                    'peaks_ms2_mz' in ref_row and 
                    'peaks_ms2_intensities' in ref_row and
                    isinstance(ref_row['peaks_ms2_mz'], (list, np.ndarray)) and
                    isinstance(ref_row['peaks_ms2_intensities'], (list, np.ndarray))
                ):
                    continue

                score = comparator.calculate_similarity_score(
                    row['peaks_mz_ms2'],
                    row['peaks_intensities_ms2'],
                    ref_row['peaks_ms2_mz'],
                    ref_row['peaks_ms2_intensities']
                )
                best_score = max(best_score, score)

            matches_df.loc[idx, 'ms2_similarity_score'] = best_score

        # Recalcul des niveaux de confiance
        print("\n📊 Calcul des niveaux de confiance...")
        for idx in tqdm(matches_df.index, desc="Attribution niveaux"):
            if not level1_mask[idx]:  # Ne pas recalculer pour le niveau 1
                confidence_level, reason = assign_confidence_level(matches_df.loc[idx])
                matches_df.loc[idx, 'confidence_level'] = confidence_level
                matches_df.loc[idx, 'confidence_reason'] = reason

        total_candidates = len(matches_df)
        unique_molecules = matches_df['match_name'].nunique()
        print(f"   ✓ {total_candidates} candidats potentiels"
              f" ({unique_molecules} molécules uniques)")

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Erreur lors du calcul des scores MS2: {str(e)}")
        raise