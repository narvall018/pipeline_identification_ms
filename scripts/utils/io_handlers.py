#scripts/utils/io_handlers.py
#-*- coding:utf-8 -*-


# Importation des modules
import re
import logging
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Tuple, Optional, Dict


# Initialiser le logger
logger = logging.getLogger(__name__)


def sanitize_filename(filename: str) -> str:
	"""
	Nettoie le nom de fichier en remplaçant les caractères problématiques.

	Args:
		filename (str): Nom du fichier à nettoyer.

	Returns:
		str: Nom de fichier nettoyé.
	"""
	# Remplace les espaces par des underscores pour éviter les problèmes liés aux espaces dans les noms de fichiers
	filename = filename.replace(' ', '_')

	# Remplace tous les caractères non alphanumériques (sauf underscores, tirets et points) par des underscores
	filename = re.sub(r'[^\w.-]', '_', filename)

	# Retourne le nom de fichier nettoyé
	return filename


def read_parquet_data(file_path: str) -> Tuple[pa.Table, Optional[Dict[bytes, bytes]]]:
	"""
	Lit un fichier Parquet et retourne les données sous forme de DataFrame et les métadonnées.

	Args:
		file_path (str): Chemin vers le fichier Parquet.

	Returns:
		Tuple[pa.Table, Optional[Dict[bytes, bytes]]]: Tuple contenant les données sous forme de `pa.Table`
		et les métadonnées du fichier.

	Raises:
		Exception: Si une erreur survient pendant la lecture du fichier.
	"""
	try:
		# Ouvre le fichier Parquet spécifié
		parquet_file = pq.ParquetFile(file_path)

		# Lit les données du fichier sous forme de tableau Apache Arrow
		table = parquet_file.read()

		# Récupère les métadonnées associées au schéma du fichier
		metadata = parquet_file.schema.metadata

		# Log le succès de l'opération
		logger.info(f"Fichier Parquet lu avec succès : {file_path}")

		# Retourne les données et les métadonnées
		return table, metadata

	except Exception as e:
		# Log l'erreur si la lecture échoue
		logger.error(f"Erreur lors de la lecture du fichier {file_path} : {str(e)}")

		# Relève une exception pour signaler le problème
		raise


def save_peaks(
	df: pa.Table,
	sample_name: str,
	step: str,
	data_type: str = 'samples',
	metadata: Optional[Dict[bytes, bytes]] = None
) -> Path:
	"""
	Sauvegarde les résultats de la détection de pics sous forme de fichier Parquet.

	Args:
		df (pa.Table): Table contenant les données à sauvegarder.
		sample_name (str): Nom de l'échantillon.
		step (str): Étape de traitement (ex : 'raw', 'processed').
		data_type (str, optional): Type de données (ex : 'samples', 'calibrants'). Par défaut : 'samples'.
		metadata (Optional[Dict[bytes, bytes]], optional): Métadonnées supplémentaires à inclure.

	Returns:
		Path: Chemin complet du fichier Parquet sauvegardé.

	Raises:
		Exception: Si une erreur survient pendant la sauvegarde.
	"""
	try:
		# Nettoie le nom de l'échantillon pour éviter des caractères problématiques
		safe_sample_name = sanitize_filename(sample_name)
		logger.info(f"Nom de l'échantillon nettoyé : {safe_sample_name}")

		# Crée le répertoire cible pour la sauvegarde des fichiers
		base_dir = Path(f"data/intermediate/{data_type}/{safe_sample_name}/ms1")
		base_dir.mkdir(parents=True, exist_ok=True)

		# Définit le chemin complet du fichier Parquet
		file_path = base_dir / f"{step}.parquet"

		# Si `df` est un DataFrame pandas, le convertit en table Arrow
		table = pa.Table.from_pandas(df, preserve_index=False)

		# Ajoute les métadonnées si elles sont fournies
		if metadata:
			table = table.replace_schema_metadata(metadata)

		# Sauvegarde la table Arrow au format Parquet
		pq.write_table(table, str(file_path))
		logger.info(f"Données sauvegardées avec succès : {file_path}")

		# Retourne le chemin du fichier sauvegardé
		return file_path

	except Exception as e:
		# Log l'erreur si la sauvegarde échoue
		logger.error(f"Erreur lors de la sauvegarde de l'étape {step} pour l'échantillon {sample_name} : {str(e)}")

		# Relève une exception pour signaler le problème
		raise
