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
    Trouve les correspondances entre les pics et une base de données de molécules dans une fenêtre définie par des tolérances.

    Args:
        peaks_df (pd.DataFrame): Données des pics détectés (m/z, RT, CCS, etc.).
        db_df (pd.DataFrame): Base de données des molécules de référence.
        tolerances (Optional[Dict[str, float]]): Tolérances pour les correspondances (m/z en ppm, CCS en %, RT en minutes).

    Returns:
        pd.DataFrame: DataFrame contenant les correspondances avec les scores et niveaux de confiance.
    """
    # Définit les tolérances par défaut si elles ne sont pas fournies
    if tolerances is None:
        tolerances = {
            'mz_ppm': 5,           # Tolérance en ppm pour la valeur m/z
            'ccs_percent': 8,      # Tolérance en pourcentage pour la CCS
            'rt_min': 2            # Tolérance en minutes pour le temps de rétention
        }

    # Création de clés composites dans `db_df` pour 'Name', 'adduct' et 'SMILES'
    db_df['Name_str'] = db_df['Name'].astype(str).fillna('')
    db_df['adduct_str'] = db_df['adduct'].astype(str).fillna('')
    db_df['SMILES_str'] = db_df['SMILES'].astype(str).fillna('')

    # Génère un identifiant unique par molécule basé sur 'Name' et 'adduct', ou sur 'SMILES' si indisponibles
    db_df['molecule_id'] = db_df.apply(
        lambda row: f"{row['Name_str']}_{row['adduct_str']}" if row['Name_str'] and row['adduct_str'] else row['SMILES_str'],
        axis=1
    )

    # Ajoute une colonne pour indiquer la disponibilité des données MS2
    db_df['has_ms2_db'] = db_df['peaks_ms2_mz'].apply(lambda x: 1 if isinstance(x, list) else 0)

    # Agrège par `molecule_id` pour vérifier si MS2 est disponible au moins une fois
    ms2_df = db_df.groupby('molecule_id')['has_ms2_db'].max().reset_index()
    db_df = db_df.drop(columns=['has_ms2_db']).merge(ms2_df, on='molecule_id', how='left')

    # Trie la base de données par m/z pour une recherche optimisée
    db_df = db_df.sort_values('mz').reset_index(drop=True)
    db_mz = db_df['mz'].values

    all_matches = []  # Stocke les correspondances trouvées

    # Parcourt chaque pic dans les données des pics détectés
    for peak in peaks_df.itertuples():
        # Calcule les limites de tolérance pour le m/z
        mz_tolerance = peak.mz * tolerances['mz_ppm'] * 1e-6
        mz_min, mz_max = peak.mz - mz_tolerance, peak.mz + mz_tolerance

        # Recherche efficace des indices correspondant aux limites de tolérance
        idx_start = np.searchsorted(db_mz, mz_min, side='left')
        idx_end = np.searchsorted(db_mz, mz_max, side='right')
        matches = db_df.iloc[idx_start:idx_end]

        if not matches.empty:
            # Parcourt chaque correspondance potentielle
            for match in matches.itertuples():
                # Vérification des tolérances pour le temps de rétention (RT)
                rt_error, rt_match = None, False
                if pd.notna(match.Observed_RT):
                    rt_error = abs(peak.retention_time - match.Observed_RT)
                    rt_match = rt_error <= tolerances['rt_min']
                elif pd.notna(match.Predicted_RT):
                    rt_error = abs(peak.retention_time - match.Predicted_RT)
                    rt_match = rt_error <= tolerances['rt_min']

                # Vérification des tolérances pour la CCS
                ccs_error, ccs_match = None, False
                if pd.notna(match.ccs_exp):
                    ccs_error = abs((peak.CCS - match.ccs_exp) / match.ccs_exp * 100)
                    ccs_match = ccs_error <= tolerances['ccs_percent']
                elif pd.notna(match.ccs_pred):
                    ccs_error = abs((peak.CCS - match.ccs_pred) / match.ccs_pred * 100)
                    ccs_match = ccs_error <= tolerances['ccs_percent']

                # Ajoute la correspondance si elle respecte les tolérances
                if (rt_match or rt_error is None) and (ccs_match or ccs_error is None):
                    match_details = {
                        'peak_mz': peak.mz,
                        'peak_rt': peak.retention_time,
                        'peak_dt': peak.drift_time,
                        'peak_intensity': peak.intensity,
                        'peak_ccs': peak.CCS,
                        'match_name': match.Name,
                        'match_adduct': match.adduct,
                        'match_smiles': match.SMILES,
                        'match_mz': match.mz,
                        'mz_error_ppm': (peak.mz - match.mz) / match.mz * 1e6,
                        'match_ccs_exp': match.ccs_exp,
                        'match_ccs_pred': match.ccs_pred,
                        'match_rt_obs': match.Observed_RT,
                        'match_rt_pred': match.Predicted_RT,
                        'rt_error_min': rt_error,
                        'ccs_error_percent': ccs_error,
                        'has_ms2_db': match.has_ms2_db,
                        'molecule_id': match.molecule_id
                    }

                    # Calcul des scores et du niveau de confiance
                    score_details = calculate_match_scores(match_details)
                    confidence_level, confidence_reason = assign_confidence_level(match_details)

                    match_details.update({
                        'individual_scores': score_details['individual_scores'],
                        'global_score': score_details['global_score'],
                        'ccs_source': score_details['ccs_source'],
                        'rt_source': score_details['rt_source'],
                        'confidence_level': confidence_level,
                        'confidence_reason': confidence_reason
                    })
                    all_matches.append(match_details)

    # Conversion des correspondances en DataFrame
    matches_df = pd.DataFrame(all_matches) if all_matches else pd.DataFrame()

    if not matches_df.empty:
        # Trie pour prioriser les correspondances avec MS2 et le score global
        matches_df = matches_df.sort_values(
            ['molecule_id', 'has_ms2_db', 'global_score'],
            ascending=[True, False, False]
        )
        # Supprime les doublons en conservant les meilleures correspondances
        matches_df = matches_df.drop_duplicates(subset='molecule_id', keep='first')

        # Trie final par niveau de confiance et score global
        matches_df = matches_df.sort_values(['confidence_level', 'global_score'], ascending=[True, False])

    return matches_df
