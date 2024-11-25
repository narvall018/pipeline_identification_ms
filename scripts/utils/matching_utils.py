#scripts/utils/matching_utils.py
#-*- coding:utf-8 -*-


# Importation des modules
import numpy as np
import pandas as pd
from ..config.config import Config
from typing import Optional, Dict, Tuple, Any


def calculate_match_scores(
	match: Dict[str, float],
	tolerances: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
	"""
	Calcule les scores individuels et globaux pour une correspondance.

	Args:
		match (Dict[str, float]): Informations sur une correspondance (m/z, CCS, RT, etc.).
		tolerances (Optional[Dict[str, float]]): Tolérances pour les erreurs m/z, CCS et RT.

	Returns:
		Dict[str, Any]: Scores individuels, score global, et sources CCS/RT.
	"""
	# Définit les tolérances par défaut si elles ne sont pas fournies
	if tolerances is None:
		tolerances = {
			'mz_ppm': 5,          # Tolérance pour l'erreur m/z (en ppm)
			'ccs_percent': 8,     # Tolérance pour l'erreur CCS (en pourcentage)
			'rt_min': 2           # Tolérance pour l'erreur RT (en minutes)
		}

	# Poids par défaut pour chaque score (doivent totaliser 1 une fois normalisés)
	weights = {'mz': 0.4, 'ccs': 0.4, 'rt': 0.2}
	scores = {}

	# Calcul du score m/z basé sur l'erreur ppm
	scores['mz'] = max(0, 1 - abs(match['mz_error_ppm']) / tolerances['mz_ppm'])

	# Calcul du score CCS
	if pd.notna(match['match_ccs_exp']):  # Si une CCS expérimentale est disponible
		ccs_error = abs(match['ccs_error_percent'])
		weights['ccs'] *= 1.2  # Augmente le poids du CCS si une valeur expérimentale est utilisée
		scores['ccs'] = max(0, 1 - ccs_error / tolerances['ccs_percent'])
	elif pd.notna(match['match_ccs_pred']):  # Si une CCS prédite est disponible
		# Calcule l'erreur CCS par rapport à la valeur prédite
		match['ccs_error_percent'] = (match['peak_ccs'] - match['match_ccs_pred']) / match['match_ccs_pred'] * 100
		ccs_error = abs(match['ccs_error_percent'])
		weights['ccs'] *= 0.6  # Réduit le poids si la CCS prédite est utilisée
		scores['ccs'] = max(0, 1 - ccs_error / tolerances['ccs_percent'])
	else:  # Si aucune information CCS n'est disponible
		scores['ccs'] = 0

	# Calcul du score RT
	if pd.notna(match['match_rt_obs']):  # Si un temps de rétention observé est disponible
		rt_error = abs(match['rt_error_min'])
		weights['rt'] *= 1.2  # Augmente le poids du RT si une valeur observée est utilisée
		scores['rt'] = max(0, 1 - rt_error / tolerances['rt_min'])
	elif pd.notna(match['match_rt_pred']):  # Si un temps de rétention prédit est disponible
		# Calcule l'erreur RT par rapport à la valeur prédite
		match['rt_error_min'] = abs(match['peak_rt'] - match['match_rt_pred'])
		rt_error = match['rt_error_min']
		weights['rt'] *= 0.6  # Réduit le poids si le RT prédit est utilisé
		scores['rt'] = max(0, 1 - rt_error / tolerances['rt_min'])
	else:  # Si aucune information RT n'est disponible
		scores['rt'] = 0

	# Normalisation des poids pour garantir qu'ils totalisent 1
	total_weight = sum(weights.values())
	weights = {k: v / total_weight for k, v in weights.items()}

	# Calcul du score global comme moyenne pondérée des scores individuels
	global_score = sum(scores[key] * weights[key] for key in weights)

	# Retourne les scores individuels, le score global, et les sources utilisées pour CCS et RT
	return {
		'individual_scores': scores,
		'global_score': global_score,
		'ccs_source': 'exp' if pd.notna(match['match_ccs_exp']) else 'pred',
		'rt_source': 'obs' if pd.notna(match['match_rt_obs']) else 'pred'
	}


def assign_confidence_level(
	match: Dict[str, Any],
	tolerances: Optional[Dict[str, float]] = None
) -> Tuple[int, str]:
	"""
	Assigne un niveau de confiance basé sur les critères de correspondance.

	Args:
		match (Dict[str, Any]): Informations sur une correspondance.
		tolerances (Optional[Dict[str, float]]): Tolérances pour les différents niveaux de confiance.

	Returns:
		Tuple[int, str]: Niveau de confiance (1-5) et raison correspondante.
	"""
	# Définit les tolérances par défaut si elles ne sont pas fournies
	if tolerances is None:
		tolerances = {
			'mz_ppm': 5,          # Tolérance pour l'erreur m/z (en ppm)
			'ccs_exp': 8,         # Tolérance pour CCS expérimentale (en pourcentage)
			'ccs_exp_l2': 8,      # Tolérance pour CCS expérimentale, niveau 2
			'rt_obs_l1': 0.5,     # Tolérance pour RT observé, niveau 1 (en minutes)
			'rt_obs_l2': 1.0,     # Tolérance pour RT observé, niveau 2 (en minutes)
			'rt_pred': 3.0        # Tolérance pour RT prédit (en minutes)
		}

	# Vérifie si l'erreur m/z est dans la tolérance
	mz_ok = abs(match['mz_error_ppm']) <= tolerances['mz_ppm']
	if not mz_ok:
		# Retourne un niveau de confiance faible si m/z est hors tolérance
		return 5, "Match m/z hors tolérance (5 ppm)"

	# Vérifie la disponibilité des données CCS et RT
	has_ccs_exp = pd.notna(match['match_ccs_exp'])
	has_ccs_pred = pd.notna(match['match_ccs_pred'])
	has_rt_obs = pd.notna(match['match_rt_obs'])
	has_rt_pred = pd.notna(match['match_rt_pred'])

	# Cas 1 : CCS expérimentale et RT observé sont disponibles
	if has_ccs_exp and has_rt_obs:
		# Vérifie les tolérances pour un match parfait
		if abs(match['ccs_error_percent']) <= tolerances['ccs_exp'] and abs(match['rt_error_min']) <= tolerances['rt_obs_l1']:
			return 1, "Match parfait (CCS exp + RT obs)"
		# Vérifie les tolérances pour un match très probable
		if abs(match['rt_error_min']) <= tolerances['rt_obs_l2']:
			return 2, "Match très probable (CCS exp + RT obs)"

	# Cas 2 : CCS expérimentale disponible avec RT observé ou prédit
	if has_ccs_exp and (has_rt_obs or has_rt_pred):
		return 3, "Match probable (CCS exp + RT disponible)"

	# Cas 3 : CCS prédite disponible
	if has_ccs_pred:
		return 4, "Match possible (CCS pred disponible)"

	# Cas 4 : Aucun CCS ou RT disponible, correspondance uniquement sur m/z
	return 5, "Match incertain (m/z uniquement)"


def find_matches_asof(
	peaks_df: pd.DataFrame,
	db_df: pd.DataFrame,
	tolerances: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
	"""
	Trouve des correspondances entre les pics et la base de données en utilisant `merge_asof`.

	Args:
		peaks_df (pd.DataFrame): Données des pics.
		db_df (pd.DataFrame): Base de données de référence.
		tolerances (Optional[Dict[str, float]]): Tolérances pour la correspondance.

	Returns:
		pd.DataFrame: Résultats des correspondances sous forme de DataFrame.
	"""
	# Utilise les tolérances par défaut si aucune n'est spécifiée
	if tolerances is None:
		tolerances = Config.IDENTIFICATION['tolerances']

	# Trie les DataFrames par 'mz' pour garantir le bon fonctionnement de `merge_asof`
	peaks_df = peaks_df.sort_values('mz')
	db_df = db_df.sort_values('mz')

	# Calcule la tolérance m/z en unités absolues (Da) à partir de la tolérance ppm
	mz_tolerance = peaks_df['mz'].mean() * tolerances['mz_ppm'] * 1e-6

	# Utilise `merge_asof` pour trouver les correspondances les plus proches en m/z
	matches = pd.merge_asof(
		peaks_df,
		db_df,
		on='mz',
		direction='nearest',  # Cherche la correspondance la plus proche en m/z
		tolerance=mz_tolerance  # Applique la tolérance en m/z
	)

	# Retourne le DataFrame contenant les correspondances
	return matches


def find_matches_window(
	peaks_df: pd.DataFrame,
	db_df: pd.DataFrame,
	tolerances: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
	"""
	Trouve les correspondances entre les pics et la base de données dans une fenêtre définie par des tolérances.

	Args:
		peaks_df (pd.DataFrame): Données des pics détectés.
		db_df (pd.DataFrame): Base de données de référence.
		tolerances (Optional[Dict[str, float]]): Tolérances pour la correspondance.

	Returns:
		pd.DataFrame: DataFrame contenant les correspondances trouvées avec les détails.
	"""
	# Définit les tolérances par défaut si elles ne sont pas fournies
	if tolerances is None:
		tolerances = {
			'mz_ppm': 5,         # Tolérance pour m/z (en ppm)
			'ccs_percent': 8,    # Tolérance pour CCS (en pourcentage)
			'rt_min': 2          # Tolérance pour RT (en minutes)
		}

	# Trie les données de la base de données par 'mz' pour les recherches optimisées
	db_df = db_df.sort_values('mz').reset_index(drop=True)
	db_mz = db_df['mz'].values  # Extrait les valeurs de m/z en tant que tableau numpy

	# Liste pour stocker toutes les correspondances trouvées
	all_matches = []

	# Parcourt chaque pic dans les données des pics détectés
	for peak in peaks_df.itertuples():
		# Calcule la tolérance en m/z en unités absolues (Da)
		mz_tolerance = peak.mz * tolerances['mz_ppm'] * 1e-6

		# Définit les limites inférieure et supérieure pour la fenêtre de recherche
		mz_min = peak.mz - mz_tolerance
		mz_max = peak.mz + mz_tolerance

		# Trouve les indices dans la base de données correspondant à la fenêtre définie
		idx_start = np.searchsorted(db_mz, mz_min, side='left')
		idx_end = np.searchsorted(db_mz, mz_max, side='right')

		# Extrait les correspondances dans la fenêtre
		matches = db_df.iloc[idx_start:idx_end]

		# Si des correspondances sont trouvées, les ajoute à la liste des résultats
		if not matches.empty:
			for match in matches.itertuples():
				# Calcule les détails de la correspondance
				match_details = {
					'peak_mz': peak.mz,
					'peak_rt': peak.retention_time,
					'peak_dt': peak.drift_time,
					'match_mz': match.mz,
					'mz_error_ppm': (peak.mz - match.mz) / match.mz * 1e6
				}
				all_matches.append(match_details)

	# Retourne les correspondances sous forme de DataFrame
	return pd.DataFrame(all_matches)
