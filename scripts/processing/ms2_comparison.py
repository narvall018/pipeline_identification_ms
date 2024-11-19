import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import List, Tuple, Dict
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)

class MS2Comparator:
    def __init__(self, tolerance_mz: float = 0.01):
        """
        Initialise le comparateur MS2
        
        Args:
            tolerance_mz: Tolérance en m/z pour la comparaison des pics (en Da)
        """
        self.tolerance_mz = tolerance_mz
        
    def normalize_spectrum(self, mz_list: List[float], intensity_list: List[float]) -> Tuple[List[float], List[float]]:
        """
        Normalise les intensités par rapport au pic le plus intense
        """
        if not isinstance(mz_list, list) or not isinstance(intensity_list, list) or not mz_list or not intensity_list:
            return [], []
            
        intensity_array = np.array(intensity_list)
        max_intensity = np.max(intensity_array)
        if max_intensity == 0:
            return [], []
        normalized_intensities = (intensity_array / max_intensity) * 1000
        
        return mz_list, normalized_intensities.tolist()
        
    def align_spectra(self, exp_mz: List[float], exp_int: List[float], 
                      ref_mz: List[float], ref_int: List[float]) -> Tuple[List[float], List[float]]:
        """
        Aligne deux spectres en fonction des m/z communs dans la tolérance donnée
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
                
        # Ajouter les pics de référence qui n'ont pas de correspondance
        for j, mz_ref in enumerate(ref_mz):
            matched = False
            for mz_exp in exp_mz:
                if abs(mz_exp - mz_ref) <= self.tolerance_mz:
                    matched = True
                    break
            if not matched:
                aligned_exp_int.append(0)
                aligned_ref_int.append(ref_int[j])
                
        return aligned_exp_int, aligned_ref_int
        
    def calculate_similarity_score(self, exp_mz: List[float], exp_int: List[float],
                                 ref_mz: List[float], ref_int: List[float]) -> float:
        """
        Calcule le score de similarité entre deux spectres
        """
        try:
            # Vérification et conversion des données
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
                
            # Calcul du score cosine
            similarity = 1 - cosine(aligned_exp, aligned_ref)
            return max(0, similarity)  # Éviter les valeurs négatives dues aux erreurs d'arrondi
            
        except Exception as e:
            logger.error(f"Erreur dans le calcul du score de similarité: {str(e)}")
            return 0.0

def add_ms2_scores(matches_file: Path, identifier) -> None:
    """
    Ajoute les scores de similarité MS2 pour tous les matches
    
    Args:
        matches_file: Chemin vers le fichier all_matches.parquet
        identifier: Instance de CompoundIdentifier contenant la base de données
    """
    try:
        # Lecture des matches
        matches_df = pd.read_parquet(matches_file)
        
        # Initialisation du comparateur
        comparator = MS2Comparator(tolerance_mz=0.01)
        
        # Pour chaque match
        for idx, row in matches_df.iterrows():
            best_score = 0.0
            
            # Vérifier si les colonnes MS2 existent et ne sont pas vides
            has_ms2_data = (
                'peaks_mz_ms2' in row.index and 
                'peaks_intensities_ms2' in row.index and 
                isinstance(row['peaks_mz_ms2'], (list, np.ndarray)) and 
                isinstance(row['peaks_intensities_ms2'], (list, np.ndarray)) and
                len(row['peaks_mz_ms2']) > 0 and 
                len(row['peaks_intensities_ms2']) > 0
            )
            
            if not has_ms2_data:
                matches_df.loc[idx, 'ms2_similarity_score'] = 0.0
                continue
                
            # Trouver tous les spectres de référence correspondants
            ref_spectra = identifier.db[
                (identifier.db['Name'] == row['match_name']) & 
                (identifier.db['adduct'] == row['match_adduct'])
            ]
            
            # Comparer avec chaque spectre de référence
            for _, ref_row in ref_spectra.iterrows():
                # Vérifier si le spectre de référence a des données MS2 valides
                if (
                    not isinstance(ref_row['peaks_ms2_mz'], (list, np.ndarray)) or 
                    not isinstance(ref_row['peaks_ms2_intensities'], (list, np.ndarray)) or
                    pd.isna(ref_row['peaks_ms2_mz']).any() or
                    pd.isna(ref_row['peaks_ms2_intensities']).any()
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
            
        # Sauvegarde des résultats
        matches_df.to_parquet(matches_file)
        
        # Log des résultats
        n_with_ms2 = (matches_df['ms2_similarity_score'] > 0).sum()
        n_total = len(matches_df)
        logger.info(f"Scores MS2 calculés pour {n_with_ms2}/{n_total} matches")
        
    except Exception as e:
        logger.error(f"Erreur lors du calcul des scores MS2: {str(e)}")
        raise
