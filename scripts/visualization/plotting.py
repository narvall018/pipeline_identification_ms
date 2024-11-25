#scripts/visualization/plotting.py
#-*- coding:utf-8 -*-


# Importation des modules
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import Union


def plot_unique_molecules_per_sample(samples_dir: Union[str, Path]) -> plt.Figure:
	"""
	Génère un graphique du nombre de molécules uniques par échantillon à partir des données de correspondance.

	Args:
		samples_dir (Union[str, Path]): Chemin vers le dossier contenant les dossiers d'échantillons.

	Returns:
		plt.Figure: Objet Matplotlib contenant le graphique généré.

	Raises:
		ValueError: Si aucune donnée de correspondance n'est trouvée.
	"""
	# Liste pour stocker les résultats
	results = []

	# Convertit le chemin en objet Path si nécessaire
	samples_path = Path(samples_dir)

	# Parcourt chaque sous-dossier dans le dossier d'échantillons
	for sample_dir in samples_path.iterdir():
		if sample_dir.is_dir():
			# Chemin vers le fichier `all_matches.parquet`
			matches_file = sample_dir / 'ms1' / 'identifications' / 'all_matches.parquet'

			if matches_file.exists():
				try:
					# Charge les données de correspondance
					matches_df = pd.read_parquet(matches_file)

					# Compte le nombre de molécules uniques
					n_unique = len(matches_df['match_name'].unique())

					# Ajoute le résultat à la liste
					results.append({
						'sample': sample_dir.name,
						'n_molecules': n_unique
					})
				except Exception as e:
					# Affiche un message d'erreur mais continue le traitement des autres fichiers
					print(f"Erreur lors du traitement de {matches_file}: {e}")

	# Vérifie si des résultats ont été collectés
	if not results:
		raise ValueError("Aucune donnée de correspondance trouvée dans le dossier spécifié.")

	# Crée un DataFrame à partir des résultats collectés
	results_df = pd.DataFrame(results)

	# Crée le graphique
	plt.figure(figsize=(12, 6))
	sns.barplot(data=results_df, x='sample', y='n_molecules', palette="viridis")

	# Ajoute le titre et les étiquettes des axes
	plt.title("Nombre de molécules uniques par échantillon")
	plt.xlabel("Échantillon")
	plt.ylabel("Nombre de molécules uniques")

	# Applique une rotation aux étiquettes des échantillons pour une meilleure lisibilité
	plt.xticks(rotation=45, ha='right')

	# Ajuste la mise en page pour éviter les chevauchements
	plt.tight_layout()

	# Retourne l'objet figure
	return plt.gcf()
