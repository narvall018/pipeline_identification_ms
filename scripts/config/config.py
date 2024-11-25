#scripts/config/config.py
#-*- coding:utf-8 -*-


# Importaion des modules
from typing import Dict, Union


class Config(object):
	"""
	Classe de configuration pour définir les chemins de données, les paramètres de détection de pics,
	de clustering, d'identification, et d'extraction MS2 dans le cadre d'une analyse spectrométrique.

	Attributes:
		INPUT_SAMPLES (str): Chemin vers les échantillons d'entrée.
		INPUT_CALIBRANTS (str): Chemin vers les calibrants d'entrée.
		INPUT_DATABASES (str): Chemin vers les bases de données d'entrée.
		INTERMEDIATE_SAMPLES (str): Chemin vers les fichiers intermédiaires des échantillons.
		INTERMEDIATE_CALIBRANTS (str): Chemin vers les fichiers intermédiaires des calibrants.
		PEAK_DETECTION (dict): Paramètres pour la détection des pics.
		CLUSTERING (dict): Paramètres pour le clustering des données.
		IDENTIFICATION (dict): Paramètres pour l'identification des molécules.
		DB_COLUMNS (dict): Mappage des colonnes de la base de données.
		MS2_EXTRACTION (dict): Paramètres pour l'extraction MS2.
	"""
	# Chemins des données
	INPUT_SAMPLES: str = "data/input/samples"
	INPUT_CALIBRANTS: str = "data/input/calibrants"
	INPUT_DATABASES: str = "data/input/databases"
	
	INTERMEDIATE_SAMPLES: str = "data/intermediate/samples"
	INTERMEDIATE_CALIBRANTS: str = "data/intermediate/calibrants"
	
	# Paramètres de détection des pics
	PEAK_DETECTION: Dict[str, Union[int, Dict[str, Union[int, float]]]] = {
		'threshold': 100,  # Intensité minimale pour considérer un pic.
		'smooth_iterations': 7,  # Nombre d'itérations pour le lissage des données.
		'smooth_radius': {  # Rayons de lissage appliqués aux dimensions analytiques.
			'mz': 0,  # m/z
			'drift_time': 1,  # Temps de dérive
			'retention_time': 0  # Temps de rétention
		},
		'peak_radius': {  # Rayons pour définir une région de détection des pics.
			'mz': 2,
			'drift_time': 10,
			'retention_time': 0
		}
	}
	
	# Paramètres de clustering
	CLUSTERING: Dict[str, Union[float, Dict[str, Union[float, int]]]] = {
		'tolerances': {
			'mz': 1e-4,  # Tolérance pour m/z (en ppm ou Da).
			'dt': 0.10,  # Tolérance pour le temps de dérive (en pourcentage).
			'rt': 0.20  # Tolérance pour le temps de rétention (en pourcentage).
		},
		'dbscan': {
			'eps': 1.0,  # Distance maximale entre deux points pour les considérer comme voisins.
			'min_samples': 2  # Nombre minimum de points pour former un cluster.
		}
	}
	
	# Paramètres d'identification
	IDENTIFICATION: Dict[str, Union[str, Dict[str, Union[int, float]]]] = {
		'database_file': "norman_all_ccs_all_rt_pos_neg_with_ms2.h5",  # Nom du fichier de la base de données.
		'database_key': 'positive',  # Clé pour sélectionner les données dans la base de données.
		'tolerances': {  # Tolérances pour l'identification.
			'mz_ppm': 5,  # Tolérance en ppm pour m/z.
			'ccs_percent': 8,  # Tolérance en pourcentage pour CCS.
			'rt_min': 2  # Tolérance en minutes pour RT.
		},
		'weights': {  # Poids pour le calcul du score global d'identification.
			'mz': 0.4,
			'ccs': 0.4,
			'rt': 0.2
		}
	}
	
	# Colonnes de la base de données
	DB_COLUMNS: Dict[str, str] = {
		'name': 'Name',  # Nom de la molécule.
		'mz': 'mz',  # Colonne pour m/z.
		'ccs_exp': 'ccs_exp',  # Colonne pour CCS expérimental.
		'ccs_pred': 'ccs_pred',  # Colonne pour CCS prédit.
		'rt_obs': 'Observed_RT',  # Temps de rétention observé.
		'rt_pred': 'Predicted_RT'  # Temps de rétention prédit.
	}
	
	# Paramètres MS2
	MS2_EXTRACTION: Dict[str, Union[float, int]] = {
		'rt_tolerance': 0.00422,  # Tolérance en minutes pour le temps de rétention.
		'dt_tolerance': 0.22,  # Tolérance en millisecondes pour le temps de dérive.
		'mz_round_decimals': 3,  # Précision des arrondis pour m/z.
		'max_peaks': 10,  # Nombre maximum de pics MS2 à extraire.
		'intensity_scale': 999  # Facteur de normalisation des intensités.
	}
