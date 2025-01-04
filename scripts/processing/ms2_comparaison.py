import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from scipy.spatial.distance import cdist
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import multiprocessing as mp
from ..utils.matching_utils import assign_confidence_level

class MS2Comparator:
    def __init__(self, tolerance_mz: float = 0.01):
        self.tolerance_mz = tolerance_mz
        self.logger = logging.getLogger(__name__)
        # Utiliser 75% des cœurs disponibles pour éviter la surcharge
        self.n_workers = max(1, int(mp.cpu_count() * 0.75))

    def normalize_spectrum(self, mz: np.ndarray, intensity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalisation vectorisée des spectres."""
        if len(mz) == 0 or len(intensity) == 0:
            return np.array([]), np.array([])
            
        max_intensity = np.max(intensity)
        if max_intensity == 0:
            return np.array([]), np.array([])
            
        return mz, (intensity / max_intensity) * 1000

    def align_spectra_vectorized(self, exp_mz: np.ndarray, exp_int: np.ndarray, 
                               ref_mz: np.ndarray, ref_int: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Alignement vectorisé des spectres."""
        if len(exp_mz) == 0 or len(ref_mz) == 0:
            return np.array([]), np.array([])

        # Créer une matrice de distance entre tous les m/z
        distances = np.abs(exp_mz[:, np.newaxis] - ref_mz)
        matches = distances <= self.tolerance_mz

        # Créer les vecteurs alignés
        aligned_exp = np.zeros(len(exp_mz) + len(ref_mz))
        aligned_ref = np.zeros(len(exp_mz) + len(ref_mz))

        # Remplir les intensités correspondantes
        exp_matches = matches.any(axis=1)
        aligned_exp[:len(exp_mz)] = np.where(exp_matches, exp_int, 0)
        
        ref_matches = matches.any(axis=0)
        aligned_ref[len(exp_mz):] = np.where(ref_matches, ref_int, 0)

        return aligned_exp, aligned_ref

    def calculate_similarity_batch(self, batch_data: Tuple[pd.Series, Dict]) -> Tuple[int, float]:
        """Calcule la similarité pour un lot de données."""
        row, ref_spectra_dict = batch_data
        
        try:
            best_score = 0.0
            exp_mz = np.array(row['peaks_mz_ms2'])
            exp_int = np.array(row['peaks_intensities_ms2'])
            
            if len(exp_mz) == 0 or len(exp_int) == 0:
                return row.name, 0.0

            # Normaliser le spectre expérimental une seule fois
            exp_mz, exp_int = self.normalize_spectrum(exp_mz, exp_int)
            
            # Récupérer les spectres de référence pré-normalisés
            cache_key = f"{row['match_name']}_{row['match_adduct']}"
            if cache_key in ref_spectra_dict:
                for ref_mz, ref_int in ref_spectra_dict[cache_key]:
                    aligned_exp, aligned_ref = self.align_spectra_vectorized(
                        exp_mz, exp_int, ref_mz, ref_int
                    )
                    if len(aligned_exp) > 0:
                        # Utiliser cdist pour le calcul vectorisé de la distance cosinus
                        similarity = 1 - cdist(
                            aligned_exp.reshape(1, -1),
                            aligned_ref.reshape(1, -1),
                            metric='cosine'
                        )[0, 0]
                        best_score = max(best_score, similarity)

            return row.name, max(0, best_score)
            
        except Exception as e:
            self.logger.error(f"Erreur dans le calcul du score de similarité : {str(e)}")
            return row.name, 0.0

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
        
        # Initialisation
        matches_df['ms2_similarity_score'] = 0.0
        level1_mask = matches_df['confidence_level'] == 1
        
        # Pré-filtrage
        matches_to_analyze = matches_df[
            (~level1_mask) &
            (matches_df['has_ms2_db'] == 1) &
            matches_df['peaks_mz_ms2'].apply(lambda x: isinstance(x, list) and len(x) > 0)
        ]
        
        n_matches_with_ms2 = len(matches_to_analyze)
        print(f"   ✓ {n_matches_with_ms2}/{len(matches_df)} matches avec MS2 à analyser")
        
        # Préparation silencieuse du cache des spectres
        ref_spectra_dict = {}
        for name, adduct in matches_to_analyze[['match_name', 'match_adduct']].drop_duplicates().values:
            ref_spectra = identifier.db[
                (identifier.db['Name'] == name) & 
                (identifier.db['adduct'] == adduct)
            ]
            
            normalized_spectra = []
            for _, ref_row in ref_spectra.iterrows():
                if not (isinstance(ref_row['peaks_ms2_mz'], (list, np.ndarray)) and 
                       isinstance(ref_row['peaks_ms2_intensities'], (list, np.ndarray))):
                    continue
                    
                ref_mz = np.array(ref_row['peaks_ms2_mz'])
                ref_int = np.array(ref_row['peaks_ms2_intensities'])
                ref_mz, ref_int = comparator.normalize_spectrum(ref_mz, ref_int)
                if len(ref_mz) > 0:
                    normalized_spectra.append((ref_mz, ref_int))
                    
            if normalized_spectra:
                ref_spectra_dict[f"{name}_{adduct}"] = normalized_spectra

        # Préparation des lots
        batch_data = [(row, ref_spectra_dict) for _, row in matches_to_analyze.iterrows()]
        
        # Traitement parallèle
        with ProcessPoolExecutor(max_workers=comparator.n_workers) as executor:
            futures = [executor.submit(comparator.calculate_similarity_batch, data) 
                      for data in batch_data]
            
            for future in tqdm(futures, total=len(batch_data), desc="Calcul scores MS2"):
                idx, score = future.result()
                matches_df.loc[idx, 'ms2_similarity_score'] = score

        # Recalcul des niveaux de confiance
        print("\n📊 Calcul des niveaux de confiance...")
        for idx in tqdm(matches_df.index, desc="Attribution niveaux"):
            if not level1_mask[idx]:
                confidence_level, confidence_reason = assign_confidence_level(matches_df.loc[idx])
                matches_df.loc[idx, ['confidence_level', 'confidence_reason']] = [confidence_level, confidence_reason] 

        total_candidates = len(matches_df)
        unique_molecules = matches_df['match_name'].nunique()
        print(f"   ✓ {total_candidates} candidats potentiels"
              f" ({unique_molecules} molécules uniques)")

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Erreur lors du calcul des scores MS2: {str(e)}")
        raise