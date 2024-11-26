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
	

def plot_level1_molecules_per_sample(samples_dir: Union[str, Path]) -> plt.Figure:
    """
    Génère un graphique du nombre de molécules uniques identifiées avec un niveau de confiance 1 par échantillon.
    
    Args:
        samples_dir (Union[str, Path]): Chemin vers le dossier contenant les dossiers d'échantillons.
    
    Returns:
        plt.Figure: Objet Matplotlib contenant le graphique généré.
        
    Raises:
        ValueError: Si aucune donnée de correspondance de niveau 1 n'est trouvée.
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
                    
                    # Filtre pour ne garder que les identifications de niveau 1
                    level1_matches = matches_df[matches_df['confidence_level'] == 1]
                    
                    # Compte le nombre de molécules uniques de niveau 1
                    n_unique_level1 = len(level1_matches['match_name'].unique())
                    
                    # Ajoute le résultat à la liste
                    results.append({
                        'sample': sample_dir.name,
                        'n_molecules_level1': n_unique_level1
                    })
                except Exception as e:
                    print(f"Erreur lors du traitement de {matches_file}: {e}")
    
    # Vérifie si des résultats ont été collectés
    if not results:
        raise ValueError("Aucune donnée de correspondance de niveau 1 trouvée dans le dossier spécifié.")
    
    # Crée un DataFrame à partir des résultats collectés
    results_df = pd.DataFrame(results)
    
    # Crée le graphique
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x='sample', y='n_molecules_level1', color='green')
    
    # Ajoute le titre et les étiquettes des axes
    plt.title("Nombre de molécules uniques identifiées avec un niveau de confiance 1 par échantillon")
    plt.xlabel("Échantillon")
    plt.ylabel("Nombre de molécules niveau 1")
    
    # Applique une rotation aux étiquettes des échantillons
    plt.xticks(rotation=45, ha='right')
    
    # Ajuste la mise en page
    plt.tight_layout()
    
    # Retourne l'objet figure
    return plt.gcf()
    
    
    
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Union, Tuple, Dict, List

def create_similarity_matrix(samples_dir: Union[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Crée une matrice de similarité entre les échantillons basée sur leurs molécules communes.
    """
    samples_molecules = {}
    samples_path = Path(samples_dir)
    
    # Collecte des molécules pour chaque échantillon
    for sample_dir in samples_path.iterdir():
        if sample_dir.is_dir():
            matches_file = sample_dir / 'ms1' / 'identifications' / 'all_matches.parquet'
            if matches_file.exists():
                try:
                    matches_df = pd.read_parquet(matches_file)
                    samples_molecules[sample_dir.name] = set(matches_df['match_name'].unique())
                except Exception as e:
                    print(f"Erreur lecture {matches_file}: {e}")
                    continue
    
    if not samples_molecules:
        raise ValueError("Aucun échantillon trouvé avec des données valides")
    
    # Création de la matrice de présence/absence
    all_molecules = sorted(set.union(*samples_molecules.values()))
    molecule_matrix = pd.DataFrame(0, 
                                 index=samples_molecules.keys(),
                                 columns=all_molecules,
                                 dtype=float)  # Spécifier float dès le début
    
    for sample, molecules in samples_molecules.items():
        molecule_matrix.loc[sample, list(molecules)] = 1.0
    
    # Calcul de la matrice de similarité
    similarity_matrix = pd.DataFrame(0.0,  # Initialiser avec des float
                                   index=molecule_matrix.index,
                                   columns=molecule_matrix.index,
                                   dtype=float)
    
    for idx1 in molecule_matrix.index:
        for idx2 in molecule_matrix.index:
            molecules1 = set(molecule_matrix.columns[molecule_matrix.loc[idx1] == 1.0])
            molecules2 = set(molecule_matrix.columns[molecule_matrix.loc[idx2] == 1.0])
            
            if molecules1 or molecules2:  # Éviter la division par zéro
                similarity = len(molecules1.intersection(molecules2)) / len(molecules1.union(molecules2)) * 100
                similarity_matrix.loc[idx1, idx2] = float(similarity)
    
    return similarity_matrix, molecule_matrix

def plot_sample_similarity_heatmap(samples_dir: Union[str, Path]) -> plt.Figure:
    """
    Génère une heatmap de similarité entre échantillons avec clustering hiérarchique.
    """
    similarity_matrix, _ = create_similarity_matrix(samples_dir)
    
    # Conversion en array numpy
    similarity_array = similarity_matrix.to_numpy(dtype=float)
    
    # Clustering hiérarchique
    linkage = hierarchy.linkage(pdist(similarity_array), method='average')
    
    plt.figure(figsize=(12, 10))
    sns.clustermap(similarity_matrix,
                  cmap='YlOrRd',
                  row_linkage=linkage,
                  col_linkage=linkage,
                  annot=True,
                  fmt='.1f',
                  vmin=0,
                  vmax=100)
    
    plt.title("Similarité entre échantillons (%)")
    return plt.gcf()


def analyze_sample_clusters(samples_dir: Union[str, Path], n_clusters: int = 3) -> Dict:
    """
    Analyse les clusters d'échantillons et identifie leurs molécules caractéristiques.
    """
    similarity_matrix, molecule_matrix = create_similarity_matrix(samples_dir)
    
    # Conversion en array numpy pour le clustering
    similarity_array = similarity_matrix.to_numpy(dtype=float)
    
    # Clustering hiérarchique
    linkage = hierarchy.linkage(pdist(similarity_array), method='average')
    clusters = hierarchy.fcluster(linkage, n_clusters, criterion='maxclust')
    
    # Ajout des labels de cluster
    molecule_matrix['Cluster'] = clusters
    
    cluster_stats = {}
    
    for cluster_id in range(1, n_clusters + 1):
        cluster_samples = molecule_matrix[molecule_matrix['Cluster'] == cluster_id]
        
        if len(cluster_samples) > 0:  # Vérification qu'il y a des échantillons dans le cluster
            # Calcul des molécules caractéristiques
            molecule_freq = cluster_samples.iloc[:, :-1].astype(float).mean()
            characteristic_molecules = molecule_freq[molecule_freq > 0.75].index.tolist()
            
            cluster_stats[f'Cluster_{cluster_id}'] = {
                'n_samples': len(cluster_samples),
                'avg_molecules_per_sample': cluster_samples.iloc[:, :-1].astype(float).sum(axis=1).mean(),
                'characteristic_molecules': characteristic_molecules,
                'samples': cluster_samples.index.tolist()
            }
    
    return cluster_stats

def plot_cluster_statistics(cluster_stats: Dict) -> plt.Figure:
    """
    Visualise les statistiques des clusters.
    """
    plt.figure(figsize=(15, 6))
    
    # Nombre d'échantillons par cluster
    plt.subplot(121)
    samples_per_cluster = [stats['n_samples'] for stats in cluster_stats.values()]
    plt.bar(cluster_stats.keys(), samples_per_cluster)
    plt.title("Nombre d'échantillons par cluster")
    plt.xticks(rotation=45)
    
    # Moyenne des molécules par cluster
    plt.subplot(122)
    avg_molecules = [stats['avg_molecules_per_sample'] for stats in cluster_stats.values()]
    plt.bar(cluster_stats.keys(), avg_molecules)
    plt.title("Moyenne de molécules par échantillon dans chaque cluster")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return plt.gcf()
  
def create_similarity_matrix_by_confidence(samples_dir: Union[str, Path], confidence_levels: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Crée une matrice de similarité entre les échantillons en ne considérant que les molécules 
    avec les niveaux de confiance spécifiés.
    
    Args:
        samples_dir: Chemin vers le dossier des échantillons
        confidence_levels: Liste des niveaux de confiance à inclure (ex: [1] ou [1,2] ou [1,2,3])
    """
    samples_molecules = {}
    samples_path = Path(samples_dir)
    
    # Collecte des molécules pour chaque échantillon
    for sample_dir in samples_path.iterdir():
        if sample_dir.is_dir():
            matches_file = sample_dir / 'ms1' / 'identifications' / 'all_matches.parquet'
            if matches_file.exists():
                try:
                    matches_df = pd.read_parquet(matches_file)
                    # Filtre par niveau de confiance
                    filtered_df = matches_df[matches_df['confidence_level'].isin(confidence_levels)]
                    samples_molecules[sample_dir.name] = set(filtered_df['match_name'].unique())
                except Exception as e:
                    print(f"Erreur lecture {matches_file}: {e}")
                    continue
    
    if not samples_molecules:
        raise ValueError("Aucun échantillon trouvé avec des données valides")
    
    # Création de la matrice de présence/absence
    all_molecules = sorted(set.union(*samples_molecules.values()))
    molecule_matrix = pd.DataFrame(0, 
                                 index=samples_molecules.keys(),
                                 columns=all_molecules,
                                 dtype=float)
    
    for sample, molecules in samples_molecules.items():
        molecule_matrix.loc[sample, list(molecules)] = 1.0
    
    # Calcul de la matrice de similarité
    similarity_matrix = pd.DataFrame(0.0,
                                   index=molecule_matrix.index,
                                   columns=molecule_matrix.index,
                                   dtype=float)
    
    for idx1 in molecule_matrix.index:
        for idx2 in molecule_matrix.index:
            molecules1 = set(molecule_matrix.columns[molecule_matrix.loc[idx1] == 1.0])
            molecules2 = set(molecule_matrix.columns[molecule_matrix.loc[idx2] == 1.0])
            
            if molecules1 or molecules2:
                similarity = len(molecules1.intersection(molecules2)) / len(molecules1.union(molecules2)) * 100
                similarity_matrix.loc[idx1, idx2] = float(similarity)
    
    return similarity_matrix, molecule_matrix

def plot_sample_similarity_heatmap_by_confidence(samples_dir: Union[str, Path], 
                                               confidence_levels: List[int],
                                               title_suffix: str = "") -> plt.Figure:
    """
    Génère une heatmap de similarité entre échantillons pour des niveaux de confiance spécifiques.
    """
    similarity_matrix, _ = create_similarity_matrix_by_confidence(samples_dir, confidence_levels)
    
    # Conversion en array numpy
    similarity_array = similarity_matrix.to_numpy(dtype=float)
    
    # Clustering hiérarchique
    linkage = hierarchy.linkage(pdist(similarity_array), method='average')
    
    plt.figure(figsize=(12, 10))
    g = sns.clustermap(similarity_matrix,
                      cmap='YlOrRd',
                      row_linkage=linkage,
                      col_linkage=linkage,
                      annot=True,
                      fmt='.1f',
                      vmin=0,
                      vmax=100)
    
    # Titre plus descriptif
    confidence_text = f"niveau{'s' if len(confidence_levels) > 1 else ''} {', '.join(map(str, confidence_levels))}"
    plt.suptitle(f"Similarité entre échantillons (%) - {confidence_text}{title_suffix}", y=1.02)
    
    return g.figure

