#scripts/processing/ms2_extraction.py
#-*- coding:utf-8 -*-


# Importation des modules
import logging
import pandas as pd
from pathlib import Path


# Initialiser le logger
logger = logging.getLogger(__name__)


def extract_ms2_for_matches(matches_df: pd.DataFrame, raw_parquet_path: str, output_dir: str) -> pd.DataFrame:
	"""
	Extrait les spectres MS2 pour chaque correspondance dans un DataFrame de correspondances et
	met à jour le fichier `all_matches.parquet` avec les données MS2.

	Args:
		matches_df (pd.DataFrame): DataFrame contenant les correspondances pour lesquelles extraire les spectres MS2.
		raw_parquet_path (str): Chemin vers le fichier brut des données MS en format Parquet.
		output_dir (str): Répertoire où sauvegarder le fichier `all_matches.parquet` mis à jour.

	Returns:
		pd.DataFrame: DataFrame des correspondances mis à jour avec les colonnes `peaks_mz_ms2` et `peaks_intensities_ms2`.

	Raises:
		Exception: Si une erreur survient pendant l'extraction ou la mise à jour des données.
	"""
	try:
		# Charge les données brutes MS depuis le fichier Parquet
		print("\n🔍 Lecture du fichier brut pour MS2...")
		raw_data = pd.read_parquet(path=raw_parquet_path)

		# Convertit la colonne 'mslevel' en entier si ce n'est pas déjà le cas
		raw_data['mslevel'] = raw_data['mslevel'].astype(int)

		# Affiche la distribution des niveaux MS présents dans les données brutes
		ms_counts = raw_data['mslevel'].value_counts().sort_index()
		print(f"   ℹ️ Distribution des niveaux MS: \n{ms_counts}")

		# Filtre uniquement les spectres MS2
		ms2_data = raw_data[raw_data['mslevel'] == 2]
		print(f"   ✓ Nombre de spectres MS2 disponibles: {len(ms2_data)}")

		# Initialisation des listes pour stocker les spectres MS2 extraits
		peaks_mz_ms2_list = []
		peaks_intensities_ms2_list = []

		print("\n🎯 Extraction des spectres MS2...")
		matches_with_ms2 = 0  # Compte les correspondances avec des données MS2
		total_matches = len(matches_df)  # Total des correspondances

		# Parcourt chaque correspondance dans le DataFrame
		for idx, match in matches_df.iterrows():
			# Filtre les spectres MS2 correspondant aux critères RT, DT, et m/z précurseur
			match_ms2 = ms2_data[
				(ms2_data['rt'] >= match['peak_rt'] - 0.00422) &
				(ms2_data['rt'] <= match['peak_rt'] + 0.00422) &
				(ms2_data['dt'] >= match['peak_dt'] - 0.22) &
				(ms2_data['dt'] <= match['peak_dt'] + 0.22)
			]

			if len(match_ms2) > 0:
				matches_with_ms2 += 1

				# Regroupe les spectres MS2 en sommant les intensités par m/z arrondi
				match_ms2['mz_rounded'] = match_ms2['mz'].round(3)
				spectrum = match_ms2.groupby('mz_rounded')['intensity'].sum().reset_index()

				# Normalise les intensités
				max_intensity = spectrum['intensity'].max()
				if max_intensity > 0:
					spectrum['intensity_normalized'] = (spectrum['intensity'] / max_intensity * 999).round(0).astype(int)

					# Sélectionne les 10 pics les plus intenses
					spectrum = spectrum.nlargest(10, 'intensity')

					# Ajoute les pics normalisés au DataFrame des correspondances
					peaks_mz_ms2_list.append(spectrum['mz_rounded'].tolist())
					peaks_intensities_ms2_list.append(spectrum['intensity_normalized'].tolist())
				else:
					# Si aucune intensité valide, ajoute des listes vides
					peaks_mz_ms2_list.append([])
					peaks_intensities_ms2_list.append([])
			else:
				# Si aucun spectre MS2 correspondant, ajoute des listes vides
				peaks_mz_ms2_list.append([])
				peaks_intensities_ms2_list.append([])

		# Ajoute les colonnes MS2 au DataFrame original des correspondances
		matches_df['peaks_mz_ms2'] = peaks_mz_ms2_list
		matches_df['peaks_intensities_ms2'] = peaks_intensities_ms2_list

		# Crée le répertoire de sortie si nécessaire
		output_dir = Path(output_dir)
		output_dir.mkdir(parents=True, exist_ok=True)

		# Sauvegarde le DataFrame mis à jour dans un fichier Parquet
		output_file = output_dir / 'all_matches.parquet'
		matches_df.to_parquet(output_file)

		# Affiche les statistiques sur les correspondances avec spectres MS2
		n_with_spectra = sum(len(mz_list) > 0 for mz_list in peaks_mz_ms2_list)
		print(f"\n   ℹ️ Résultats de l'extraction MS2:")
		print(f"      - {n_with_spectra}/{total_matches} matches ont des spectres MS2 ({n_with_spectra/total_matches*100:.1f}%)")
		print(f"   ✓ Fichier all_matches.parquet mis à jour avec les spectres MS2")

		return matches_df

	except Exception as e:
		# Log l'erreur et affiche un message d'échec
		logger.error(f"Erreur lors de l'extraction MS2 : {str(e)}")
		print(f"   ✗ Erreur : {str(e)}")
		raise
