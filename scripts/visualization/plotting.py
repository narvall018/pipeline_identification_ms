#scripts/visualization/plotting.py
#-*- coding:utf-8 -*-


# Importation des modules
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Union, Tuple, Dict, List


def get_unique_molecules(matches_df: pd.DataFrame) -> set:
    """
    Extrait les noms de molécules uniques d'un DataFrame de matches,
    en ignorant les différents adduits.
    """
    return set(matches_df['match_name'].unique())

def plot_unique_molecules_per_sample(samples_dir: Union[str, Path]) -> plt.Figure:
    """
    Génère un graphique du nombre de molécules uniques par échantillon,
    en ne comptant qu'une fois chaque molécule même si elle est détectée avec différents adduits.
    """
    results = []
    samples_path = Path(samples_dir)

    for sample_dir in samples_path.iterdir():
        if sample_dir.is_dir():
            matches_file = sample_dir / 'ms1' / 'identifications' / 'all_matches.parquet'
            if matches_file.exists():
                try:
                    matches_df = pd.read_parquet(matches_file)
                    # Compte unique par nom de molécule seulement
                    unique_molecules = get_unique_molecules(matches_df)
                    results.append({
                        'sample': sample_dir.name,
                        'n_molecules': len(unique_molecules)
                    })
                except Exception as e:
                    print(f"Erreur lors du traitement de {matches_file}: {e}")

    if not results:
        raise ValueError("Aucune donnée de correspondance trouvée.")

    results_df = pd.DataFrame(results)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x='sample', y='n_molecules', palette="viridis")

    plt.title("Nombre de molécules uniques par échantillon\n(adduits multiples comptés une seule fois)")
    plt.xlabel("Échantillon")
    plt.ylabel("Nombre de molécules uniques")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    return plt.gcf()

def plot_level1_molecules_per_sample(samples_dir: Union[str, Path]) -> plt.Figure:
    """
    Graphique des molécules niveau 1 uniques par échantillon,
    en ne comptant qu'une fois chaque molécule même si elle est détectée avec différents adduits.
    """
    results = []
    samples_path = Path(samples_dir)

    for sample_dir in samples_path.iterdir():
        if sample_dir.is_dir():
            matches_file = sample_dir / 'ms1' / 'identifications' / 'all_matches.parquet'
            if matches_file.exists():
                try:
                    matches_df = pd.read_parquet(matches_file)
                    level1_matches = matches_df[matches_df['confidence_level'] == 1]
                    # Compte unique par nom de molécule pour niveau 1
                    unique_molecules = get_unique_molecules(level1_matches)
                    results.append({
                        'sample': sample_dir.name,
                        'n_molecules_level1': len(unique_molecules)
                    })
                except Exception as e:
                    print(f"Erreur lors du traitement de {matches_file}: {e}")

    if not results:
        raise ValueError("Aucune donnée de correspondance niveau 1 trouvée.")

    results_df = pd.DataFrame(results)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x='sample', y='n_molecules_level1', color='green')

    plt.title("Nombre de molécules uniques niveau 1 par échantillon\n(adduits multiples comptés une seule fois)")
    plt.xlabel("Échantillon")
    plt.ylabel("Nombre de molécules niveau 1")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    return plt.gcf()
    
    


def create_similarity_matrix(samples_dir: Union[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Crée une matrice de similarité basée sur les molécules uniques communes,
    en ignorant les différents adduits.
    """
    samples_molecules = {}
    samples_path = Path(samples_dir)

    for sample_dir in samples_path.iterdir():
        if sample_dir.is_dir():
            matches_file = sample_dir / 'ms1' / 'identifications' / 'all_matches.parquet'
            if matches_file.exists():
                try:
                    matches_df = pd.read_parquet(matches_file)
                    # Utilise seulement les noms de molécules uniques
                    samples_molecules[sample_dir.name] = get_unique_molecules(matches_df)
                except Exception as e:
                    print(f"Erreur lecture {matches_file}: {e}")
                    continue

    if not samples_molecules:
        raise ValueError("Aucun échantillon trouvé avec des données valides")

    # Création matrice présence/absence
    all_molecules = sorted(set.union(*samples_molecules.values()))
    molecule_matrix = pd.DataFrame(0,
                                 index=samples_molecules.keys(),
                                 columns=all_molecules,
                                 dtype=float)

    for sample, molecules in samples_molecules.items():
        molecule_matrix.loc[sample, list(molecules)] = 1.0

    # Calcul matrice de similarité
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
    Crée une matrice de similarité pour les niveaux de confiance spécifiés,
    en ne comptant qu'une fois chaque molécule même si elle est détectée avec différents adduits.
    """
    samples_molecules = {}
    samples_path = Path(samples_dir)

    for sample_dir in samples_path.iterdir():
        if sample_dir.is_dir():
            matches_file = sample_dir / 'ms1' / 'identifications' / 'all_matches.parquet'
            if matches_file.exists():
                try:
                    matches_df = pd.read_parquet(matches_file)
                    filtered_df = matches_df[matches_df['confidence_level'].isin(confidence_levels)]
                    # Utilise seulement les noms de molécules uniques
                    samples_molecules[sample_dir.name] = get_unique_molecules(filtered_df)
                except Exception as e:
                    print(f"Erreur lecture {matches_file}: {e}")
                    continue

    if not samples_molecules:
        raise ValueError("Aucun échantillon trouvé avec des données valides")

    # Même logique que create_similarity_matrix pour le reste
    all_molecules = sorted(set.union(*samples_molecules.values()))
    molecule_matrix = pd.DataFrame(0,
                                 index=samples_molecules.keys(),
                                 columns=all_molecules,
                                 dtype=float)

    for sample, molecules in samples_molecules.items():
        molecule_matrix.loc[sample, list(molecules)] = 1.0

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
    
    confidence_text = f"niveau{'s' if len(confidence_levels) > 1 else ''} {', '.join(map(str, confidence_levels))}"
    plt.suptitle(f"Similarité entre échantillons (%) - {confidence_text}{title_suffix}", y=1.02)
    
    return g.figure



def plot_level1_molecule_distribution_bubble(samples_dir: Union[str, Path], top_n: int = 20) -> plt.Figure:
    """
    Crée un bubble plot montrant la distribution des molécules de niveau de confiance 1 dans les échantillons.
    La taille et la couleur des bulles représentent l'intensité relative.

    Args:
        samples_dir: Chemin vers le dossier contenant les échantillons
        top_n: Nombre des molécules les plus fréquentes à afficher

    Returns:
        plt.Figure: La figure matplotlib générée
    """
    # Collecte des données
    data = []
    for sample_dir in Path(samples_dir).iterdir():
        if sample_dir.is_dir():
            matches_file = sample_dir / 'ms1' / 'identifications' / 'all_matches.parquet'
            if matches_file.exists():
                try:
                    matches_df = pd.read_parquet(matches_file)
                    # Filtrer pour niveau de confiance 1
                    level1_df = matches_df[matches_df['confidence_level'] == 1]
                    # Garder seulement les intensités maximales par molécule (ignorer les adduits)
                    sample_data = level1_df.groupby('match_name')['peak_intensity'].max().reset_index()
                    sample_data['sample'] = sample_dir.name
                    data.append(sample_data)
                except Exception as e:
                    print(f"Erreur lors du traitement de {matches_file}: {e}")
                    continue

    if not data:
        raise ValueError("Aucune donnée trouvée")

    # Combine tous les échantillons
    combined_df = pd.concat(data)
    
    # Sélectionne les molécules les plus fréquentes
    top_molecules = (combined_df.groupby('match_name')
                    .size()
                    .sort_values(ascending=False)
                    .head(top_n)
                    .index)
    
    filtered_df = combined_df[combined_df['match_name'].isin(top_molecules)]

    # Création du plot
    plt.figure(figsize=(20, 10))
    
    # Création du scatter plot
    scatter = plt.scatter(filtered_df['match_name'], 
                         filtered_df['sample'],
                         s=filtered_df['peak_intensity']/filtered_df['peak_intensity'].max()*1000,  # Taille normalisée
                         c=filtered_df['peak_intensity'],  # Couleur basée sur l'intensité
                         cmap='viridis',
                         alpha=0.6)

    # Personnalisation
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlabel('Molécules')
    plt.ylabel('Échantillons')
    plt.title('Distribution des molécules de niveau 1 par échantillon\nTaille et couleur des bulles = intensité relative')

    # Ajout d'une barre de couleur
    plt.colorbar(scatter, label='Intensité')

    # Ajustement de la mise en page
    plt.tight_layout()

    return plt.gcf()
    
    
    
    
############################################## TIC


import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
from typing import Union, List, Dict

def calculate_tic(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule le TIC pour les données MS1.
    
    Args:
        data: DataFrame avec les colonnes 'mslevel', 'rt', 'scanid', 'intensity'
    Returns:
        DataFrame avec les colonnes 'rt', 'scanid', 'intensity' (somme)
    """
    tic = (data[data['mslevel'] == "1"]
           .groupby(['rt', 'scanid'])['intensity']
           .sum()
           .reset_index()
           .sort_values('rt'))
    
    return tic

def plot_tics_interactive(samples_dir: Union[str, Path], output_dir: Union[str, Path]) -> None:
    """
    Crée un plot interactif des TIC MS1 pour tous les échantillons détectés.
    S'adapte au nombre d'échantillons présents.
    """
    # Dictionnaire pour stocker les TICs
    tics = {}
    
    # Calculer le TIC pour chaque fichier
    for file_path in Path(samples_dir).glob("*.parquet"):
        try:
            data = pd.read_parquet(file_path)
            tic = calculate_tic(data)
            tics[file_path.stem] = tic
            print(f"TIC calculé pour {file_path.stem}")
        except Exception as e:
            print(f"Erreur lors du traitement de {file_path.name}: {str(e)}")
            continue
    
    if not tics:
        print("Aucun TIC n'a pu être calculé")
        return
        
    # Création du plot
    fig = go.Figure()
    
    # Ajouter chaque TIC
    for sample_name, tic_data in tics.items():
        fig.add_trace(
            go.Scatter(
                x=tic_data['rt'],
                y=tic_data['intensity'],
                name=sample_name,
                mode='lines',
                line=dict(width=1),
                hovertemplate=(
                    f"<b>{sample_name}</b><br>" +
                    "RT: %{x:.2f} min<br>" +
                    "Intensité: %{y:,.0f}<br>" +
                    "<extra></extra>"
                )
            )
        )
    
    # Mise en page
    fig.update_layout(
        title="TIC MS1",
        xaxis_title="Temps de rétention (min)",
        yaxis_title="Intensité",
        showlegend=True,
        template='plotly_white',
        plot_bgcolor='white',
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=1.1,
            xanchor="right",
            x=1.0,
            bgcolor="white",
            bordercolor="white"
        ),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgb(240, 240, 240)',
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgb(240, 240, 240)',
            zeroline=False
        ),
        width=1200,
        height=600
    )
    
    # Sauvegarder les résultats
    output_path = Path(output_dir) / "tic_ms1_comparison.html"
    fig.write_html(output_path)
    
    static_path = Path(output_dir) / "tic_ms1_comparison.png"
    fig.write_image(static_path)
    
    print(f"TIC MS1 sauvegardés dans :\n{output_path}\n{static_path}")