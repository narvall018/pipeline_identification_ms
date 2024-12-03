#scripts/processing/ms2_comparaison.py
#-*- coding:utf-8 -*-


# Importation des modules
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from scipy.spatial.distance import cosine
from ..utils.matching_utils import assign_confidence_level 

# Initialiser le logger
logger = logging.getLogger(__name__)


class MS2Comparator(object):
	"""
	Classe pour comparer des spectres MS2 avec une tolérance donnée.

	Attributes:
		tolerance_mz (float): Tolérance en m/z pour la comparaison des pics (en Da).
	"""
	def __init__(self, tolerance_mz: float = 0.01) -> "MS2Comparator":
		"""
		Initialise le comparateur MS2.

		Args:
			tolerance_mz (float): Tolérance en m/z pour la comparaison des pics (en Da).

		Returns:
			MS2Comparator: Un objet de la classe MS2Comparator.
		"""
		# Initialise la tolérance m/z utilisée pour la comparaison des pics
		self.tolerance_mz: float = tolerance_mz


	def normalize_spectrum(self, mz_list: List[float], intensity_list: List[float]) -> Tuple[List[float], List[float]]:
		"""
		Normalise les intensités d'un spectre par rapport au pic le plus intense.

		Args:
			mz_list (List[float]): Liste des m/z.
			intensity_list (List[float]): Liste des intensités correspondantes.

		Returns:
			Tuple[List[float], List[float]]: Liste des m/z et intensités normalisées, ou des listes vides si les entrées sont invalides.
		"""
		# Vérifie si les listes de m/z ou d'intensités sont vides
		if not mz_list or not intensity_list:
			# Retourne des listes vides si l'une des deux listes est absente
			return [], []

		# Convertit la liste des intensités en tableau numpy pour les calculs
		intensity_array = np.array(intensity_list)

		# Calcule l'intensité maximale dans le tableau des intensités
		max_intensity = np.max(intensity_array)

		# Vérifie si l'intensité maximale est égale à zéro
		if max_intensity == 0:
			# Retourne des listes vides si toutes les intensités sont nulles
			return [], []

		# Normalise les intensités en fonction de l'intensité maximale
		normalized_intensities = (intensity_array / max_intensity) * 1000

		# Retourne le tuple contenant la liste des m/z et les intensités normalisées
		return mz_list, normalized_intensities.tolist()


	def align_spectra(
		self,
		exp_mz: List[float],
		exp_int: List[float],
		ref_mz: List[float],
		ref_int: List[float]
	) -> Tuple[List[float], List[float]]:
		"""
		Aligne deux spectres en fonction des m/z communs dans une tolérance donnée.

		Args:
			exp_mz (List[float]): Liste des m/z expérimentaux.
			exp_int (List[float]): Liste des intensités expérimentales.
			ref_mz (List[float]): Liste des m/z de référence.
			ref_int (List[float]): Liste des intensités de référence.

		Returns:
			Tuple[List[float], List[float]]: Intensités alignées pour les spectres expérimental et de référence.
		"""
		# Vérifie si les m/z expérimentaux ou les m/z de référence sont vides
		if not exp_mz or not ref_mz:
			# Retourne un tuple de listes vides si l'une des listes est absente
			return [], []

		# Initialise les listes pour stocker les intensités alignées
		aligned_exp_int = []
		aligned_ref_int = []

		# Parcourt chaque m/z expérimental
		for i, mz_exp in enumerate(exp_mz):
			# Initialise une variable pour indiquer si un match a été trouvé
			matched = False

			# Parcourt chaque m/z de référence
			for j, mz_ref in enumerate(ref_mz):
				# Vérifie si les m/z sont dans la tolérance définie
				if abs(mz_exp - mz_ref) <= self.tolerance_mz:
					# Ajoute les intensités correspondantes aux listes alignées
					aligned_exp_int.append(exp_int[i])
					aligned_ref_int.append(ref_int[j])

					# Indique qu'un match a été trouvé et sort de la boucle interne
					matched = True
					break

			# Si aucun match n'a été trouvé pour le m/z expérimental
			if not matched:
				# Ajoute l'intensité expérimentale et une intensité de référence nulle
				aligned_exp_int.append(exp_int[i])
				aligned_ref_int.append(0)

		# Parcourt les m/z de référence non alignés
		for j, mz_ref in enumerate(ref_mz):
			# Vérifie si le m/z de référence n'a pas de correspondance dans les m/z expérimentaux
			if all(abs(mz_exp - mz_ref) > self.tolerance_mz for mz_exp in exp_mz):
				# Ajoute une intensité expérimentale nulle et l'intensité de référence
				aligned_exp_int.append(0)
				aligned_ref_int.append(ref_int[j])

		# Retourne les intensités alignées pour les spectres expérimental et de référence
		return aligned_exp_int, aligned_ref_int


	def calculate_similarity_score(
		self, exp_mz: List[float],
		exp_int: List[float],
		ref_mz: List[float],
		ref_int: List[float]
	) -> float:
		"""
		Calcule le score de similarité entre deux spectres en utilisant la distance cosinus.

		Args:
			exp_mz (List[float]): Liste des m/z expérimentaux.
			exp_int (List[float]): Liste des intensités expérimentales.
			ref_mz (List[float]): Liste des m/z de référence.
			ref_int (List[float]): Liste des intensités de référence.

		Returns:
			float: Score de similarité (entre 0 et 1). Retourne 0 en cas d'erreur ou de données insuffisantes.
		"""
		try:
			# Vérification et conversion des données d'entrée en listes
			exp_mz = exp_mz if isinstance(exp_mz, list) else ([] if pd.isna(exp_mz).any() else exp_mz.tolist())
			exp_int = exp_int if isinstance(exp_int, list) else ([] if pd.isna(exp_int).any() else exp_int.tolist())
			ref_mz = ref_mz if isinstance(ref_mz, list) else ([] if pd.isna(ref_mz).any() else ref_mz.tolist())
			ref_int = ref_int if isinstance(ref_int, list) else ([] if pd.isna(ref_int).any() else ref_int.tolist())

			# Vérifie si les spectres contiennent des données valides
			if not exp_mz or not ref_mz:
				return 0.0

			# Normalisation des spectres expérimentaux et de référence
			exp_mz, exp_int = self.normalize_spectrum(exp_mz, exp_int)
			ref_mz, ref_int = self.normalize_spectrum(ref_mz, ref_int)

			# Vérifie si les spectres normalisés contiennent des intensités valides
			if not exp_int or not ref_int:
				return 0.0

			# Alignement des spectres pour une comparaison directe
			aligned_exp, aligned_ref = self.align_spectra(exp_mz, exp_int, ref_mz, ref_int)

			# Vérifie si l'alignement a produit des résultats
			if not aligned_exp or not aligned_ref:
				return 0.0

			# Calcul du score de similarité en utilisant la distance cosinus
			similarity = 1 - cosine(aligned_exp, aligned_ref)

			# Retourne le score en s'assurant qu'il est positif (éviter les erreurs d'arrondi)
			return max(0, similarity)

		except Exception as e:
			# Log toute erreur rencontrée pendant le calcul
			logger.error(f"Erreur dans le calcul du score de similarité : {str(e)}")
			return 0.0


def add_ms2_scores(matches_df: pd.DataFrame, identifier: object) -> None:
    """
    Ajoute les scores de similarité MS2 et recalcule les niveaux de confiance.
    """
    try:
        print("\n🔬 Analyse des spectres MS2...")
        comparator = MS2Comparator(tolerance_mz=0.01)
        from tqdm import tqdm
        
        # Pré-filtrer les matches qui nécessitent une analyse MS2
        matches_to_analyze = matches_df[
            (matches_df['has_ms2_db'] == 1) &  # Uniquement ceux avec MS2 dans la DB
            matches_df['peaks_mz_ms2'].apply(lambda x: isinstance(x, list) and len(x) > 0)  # Et qui ont des spectres exp
        ]
        
        n_matches_with_ms2 = len(matches_to_analyze)
        print(f"   ✓ {n_matches_with_ms2}/{len(matches_df)} matches avec MS2 à analyser (spectres exp + DB)")
        
        # Initialiser tous les scores à 0
        matches_df['ms2_similarity_score'] = 0.0
        
        # Créer un cache pour les spectres de référence
        ms2_ref_cache = {}
        
        # Traiter uniquement les matches sélectionnés
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
            confidence_level, reason = assign_confidence_level(matches_df.loc[idx])
            matches_df.loc[idx, 'confidence_level'] = confidence_level
            matches_df.loc[idx, 'confidence_reason'] = reason

        # Statistiques finales
        n_with_ms2_score = (matches_df['ms2_similarity_score'] > 0.2).sum()  # Seuil significatif
        n_level_1 = (matches_df['confidence_level'] == 1).sum()
        print(f"\n   ✓ {n_with_ms2_score}/{len(matches_df)} matches avec MS2 validés")
        print(f"   ✓ {n_level_1}/{len(matches_df)} matches niveau 1")

    except Exception as e:
        logger.error(f"Erreur lors du calcul des scores MS2: {str(e)}")
        raise