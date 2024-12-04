#scripts/processing/ccs_calibration.py
#-*- coding:utf-8 -*-


# Importation des modules
import logging
import numpy as np
import pandas as pd
import deimos as dm
from typing import Any


# Initialiser le logger
logger = logging.getLogger(__name__)


class CCSCalibrator(object):
	"""
	Classe pour la calibration et le calcul des valeurs CCS.

	Attributes:
		calibration_data (pd.DataFrame): Données de calibration CCS.
		calibration_model (Any): Modèle de calibration CCS après ajustement.
	"""
	def __init__(self, calibration_file: str) -> "CCSCalibrator":
		"""
		Initialise le calibrateur CCS.

		Args:
			calibration_file (str): Chemin vers le fichier de calibration CCS.

		Returns:
			CCSCalibrator: Un objet de la classe CCSCalibrator.
		"""
		# Charge les données de calibration à partir du fichier spécifié
		self.calibration_data: pd.DataFrame = self._load_calibration_data(file_path=calibration_file)

		# Initialise le modèle de calibration à None
		self.calibration_model: Any = None


	def _load_calibration_data(self, file_path: str) -> pd.DataFrame:
		"""
		Charge et prépare les données de calibration à partir d'un fichier CSV.

		Args:
			file_path (str): Chemin vers le fichier CSV contenant les données de calibration.

		Returns:
			pd.DataFrame: Données de calibration préparées.

		Raises:
			Exception: Si le fichier ne peut pas être chargé ou traité correctement.
		"""
		try:
			# Charger les données depuis le fichier CSV spécifié
			df = pd.read_csv(filepath_or_buffer=file_path)

			# Vérifie si la colonne 'CCS' est absente et si la colonne 'Reference rCCS' est disponible
			if 'CCS' not in df.columns and 'Reference rCCS' in df.columns:
				# Ajoute un message d'information dans les logs indiquant que les CCS seront calculées
				logger.info("Calcul des CCS à partir des valeurs rCCS")

				# Calcule la masse des ions en fonction du rapport m/z et de la charge
				df['Mi'] = df['Reference m/z'] * df['z']

				# Calcule la masse réduite en fonction de la masse des ions et de la masse d'une molécule d'azote
				df['mu'] = (df['Mi'] * 28.013) / (df['Mi'] + 28.013)

				# Calcule la racine carrée de la masse réduite
				df['sqrt_mu'] = np.sqrt(df['mu'])

				# Calcule la colonne CCS en fonction des rCCS et d'autres facteurs
				df['CCS'] = df['Reference rCCS'] * np.sqrt(1 / df['mu']) * df["z"]

			# Ajoute un message d'information dans les logs confirmant le chargement des données
			logger.info("Données de calibration chargées avec succès.")
			
			# Retourne le DataFrame contenant les données préparées
			return df

		except Exception as e:
			# Log l'erreur rencontrée lors du chargement ou du traitement des données
			logger.error(f"Erreur lors du chargement des données de calibration: {str(e)}")
			
			# Relève une exception pour signaler le problème
			raise


	def calibrate(self) -> None:
		"""
		Effectue la calibration CCS en ajustant un modèle à partir des données fournies.

		Args:
			None

		Returns:
			None

		Raises:
			Exception: Si une erreur survient lors de la calibration.
		"""
		
		try:
			# Ajoute un message d'information dans les logs indiquant le début de la calibration
			logger.info("Début de la calibration CCS.")

			# Effectue la calibration en ajustant un modèle avec les données mesurées
			self.calibration_model = dm.calibration.calibrate_ccs(
				mz=self.calibration_data['Measured m/z'],  # m/z mesurés
				ta=self.calibration_data['Measured Time'],  # Temps mesuré
				ccs=self.calibration_data['CCS'],  # Valeurs de CCS fournies
				q=self.calibration_data['z'],  # Charge des ions
				buffer_mass=28.013,  # Masse molaire de l'azote utilisé comme gaz tampon
				power=True  # Indique d'effectuer un ajustement basé sur une relation de puissance
			)

			# Ajoute un message d'information dans les logs confirmant la fin de la calibration
			logger.info("Calibration CCS terminée avec succès.")

		except Exception as e:
			# Log l'erreur rencontrée lors du processus de calibration
			logger.error(f"Erreur lors de la calibration: {str(e)}")
			
			# Relève une exception pour signaler le problème
			raise


	def calculate_ccs(self, peaks_df: pd.DataFrame) -> pd.DataFrame:
		"""
		Calcule les valeurs CCS pour un ensemble de pics fournis dans un DataFrame.

		Args:
			peaks_df (pd.DataFrame): DataFrame contenant les colonnes `mz` (m/z) et `drift_time`.

		Returns:
			pd.DataFrame: DataFrame avec une colonne supplémentaire `CCS` contenant les valeurs calculées.

		Raises:
			Exception: Si une erreur survient lors du calcul des CCS.
		"""

		try:
			# Vérifie si un modèle de calibration est disponible, sinon lance la calibration
			if self.calibration_model is None:
				self.calibrate()

			# Crée une copie du DataFrame d'entrée pour préserver les données originales
			df = peaks_df.copy()

			# Récupère la charge par défaut à partir des données de calibration
			default_charge = self.calibration_data['z'].iloc[0]

			# Applique le modèle de calibration pour calculer les CCS pour chaque ligne
			df['CCS'] = df.apply(
				lambda row: self.calibration_model.arrival2ccs(
					mz=row['mz'],  # m/z du pic
					ta=row['drift_time'],  # Temps de dérive du pic
					q=default_charge  # Charge par défaut
				),
				axis=1
			)

			logger.info(f"CCS calculées pour {len(df)} pics.")

			# Retourne le DataFrame avec la colonne CCS ajoutée
			return df
		except Exception as e:
			# Log l'erreur rencontrée lors du calcul des CCS
			logger.error(f"Erreur lors du calcul des CCS: {str(e)}")			
			raise
