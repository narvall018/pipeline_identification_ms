#scripts/processing/identification.py
#-*- coding:utf-8 -*-


# Importation des modules
import logging
import pandas as pd
from pathlib import Path
from typing import Optional
from ..config.config import Config
from ..utils.matching_utils import find_matches_window


# Initialiser le logger
logger = logging.getLogger(__name__)


class CompoundIdentifier(object):
	"""
	Classe pour identifier les composés à partir de données de pics en utilisant une base de données.

	Attributes:
		db (pd.DataFrame): Base de données des composés chargée en mémoire.
	"""
	def __init__(self) -> "CompoundIdentifier":
		"""
		Initialise le processus d'identification des composés en chargeant la base de données.

		Returns:
			CompoundIdentifier: Un objet de la classe CompoundIdentifier.
		"""

		# Initialise la base de données en tant que DataFrame vide
		self.db: pd.DataFrame = pd.DataFrame()

		# Charge la base de données en appelant la méthode dédiée
		self.load_database()


	def load_database(self) -> None:
		"""
		Charge la base de données à partir du fichier HDF5 défini dans la configuration.

		Raises:
			Exception: Si une erreur survient lors du chargement de la base de données.
		"""
		try:
			# Construit le chemin complet vers le fichier de base de données HDF5
			db_path = Path(Config.INPUT_DATABASES) / Config.IDENTIFICATION['database_file']

			# Charge les données HDF5 en utilisant la clé spécifiée dans la configuration
			self.db = pd.read_hdf(path_or_buf=db_path, key=Config.IDENTIFICATION['database_key'])

			# Ajoute un message dans les logs indiquant le succès du chargement et le nombre de composés
			logger.info(f"Base de données chargée avec succès : {len(self.db)} composés.")

		except Exception as e:
			# Log l'erreur rencontrée lors du chargement de la base de données
			logger.error(f"Erreur lors du chargement de la base de données : {str(e)}")

			# Relève une exception pour signaler le problème
			raise


	def identify_compounds(self, peaks_df: pd.DataFrame, output_dir: str) -> Optional[pd.DataFrame]:
		"""
		Identifie les composés correspondants pour un ensemble de pics donnés.

		Args:
			peaks_df (pd.DataFrame): DataFrame contenant les pics à identifier.
			output_dir (str): Chemin vers le répertoire où sauvegarder les résultats.

		Returns:
			Optional[pd.DataFrame]: DataFrame contenant les correspondances trouvées, ou `None` si aucun match n'est trouvé.

		Raises:
			Exception: Si une erreur survient lors de l'identification ou de la sauvegarde des résultats.
		"""
		# Ajoute un message d'information dans les logs indiquant le début du processus
		logger.info("Début du processus d'identification des composés.")

		try:
			# Effectue la recherche des correspondances dans la base de données
			matches_df = find_matches_window(peaks_df=peaks_df, db_df=self.db)

			# Vérifie si des correspondances ont été trouvées
			if matches_df.empty:
				# Ajoute un message d'avertissement dans les logs si aucun match n'est trouvé
				logger.warning("Aucune correspondance trouvée.")
				return None

			# Crée le chemin vers le répertoire de sortie
			output_dir_path = Path(output_dir)

			# Crée le répertoire de sortie si celui-ci n'existe pas encore
			output_dir_path.mkdir(parents=True, exist_ok=True)

			# Définit le chemin du fichier parquet pour sauvegarder les résultats
			matches_path = output_dir_path / 'all_matches.parquet'

			# Sauvegarde les correspondances trouvées dans un fichier parquet
			matches_df.to_parquet(path=matches_path)

			# Ajoute un message dans les logs confirmant la sauvegarde des correspondances
			logger.info(f"Correspondances sauvegardées avec succès dans : {matches_path}")

			# Retourne le DataFrame contenant les correspondances
			return matches_df

		except Exception as e:
			# Log l'erreur rencontrée lors du processus d'identification
			logger.error(f"Erreur lors de l'identification des composés : {str(e)}")

			# Relève une exception pour signaler le problème
			raise
