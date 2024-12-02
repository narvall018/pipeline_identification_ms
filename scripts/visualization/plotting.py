# scripts/visualization/plotting.py
# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
import plotly.graph_objects as go
from pathlib import Path
from typing import Union, Tuple, Dict, List
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def get_molecules_per_sample(identifications: pd.DataFrame, feature_info: pd.DataFrame, confidence_level: int = None) -> pd.DataFrame:
    """
    Compte les molécules uniques par échantillon en tenant compte des features présentes dans plusieurs échantillons.
    
    Args:
        identifications: DataFrame des identifications
        feature_info: DataFrame des infos des features
        confidence_level: Niveau de confiance à filtrer (optionnel)
        
    Returns:
        DataFrame avec le compte des molécules par échantillon
    """
    # Fusionner les données
    merged_df = pd.merge(
        identifications,
        feature_info[['feature_id', 'samples', 'source_sample']],
        left_on='feature_idx',
        right_index=True,
        how='left'
    )
    
    # Filtrer par niveau de confiance si spécifié
    if confidence_level is not None:
        merged_df = merged_df[merged_df['confidence_level'] == confidence_level]
    
    # Créer un DataFrame avec une ligne par couple molécule-échantillon
    all_sample_molecules = []
    for _, row in merged_df.iterrows():
        samples = row['samples'].split(',')
        for sample in samples:
            all_sample_molecules.append({
                'sample': sample,
                'molecule': row['match_name']
            })
    
    df_expanded = pd.DataFrame(all_sample_molecules)
    
    # Compter les molécules uniques par échantillon
    molecule_counts = df_expanded.groupby('sample')['molecule'].nunique().reset_index()
    molecule_counts.columns = ['sample', 'n_molecules']
    
    return molecule_counts

def plot_unique_molecules_per_sample(output_dir: Union[str, Path]) -> plt.Figure:
    """
    Génère un graphique du nombre de molécules uniques par échantillon.
    """
    try:
        # Lire les fichiers
        identifications = pd.read_parquet(Path(output_dir) / "feature_matrix/feature_identifications.parquet")
        feature_info = pd.read_parquet(Path(output_dir) / "feature_matrix/feature_info.parquet")
        
        # Obtenir les comptes
        molecule_counts = get_molecules_per_sample(identifications, feature_info)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=molecule_counts, x='sample', y='n_molecules', palette="viridis")
        plt.title("Nombre de molécules uniques par échantillon")
        plt.xlabel("Échantillon")
        plt.ylabel("Nombre de molécules uniques")
        
        # Ajouter les valeurs sur les barres
        for i, v in enumerate(molecule_counts['n_molecules']):
            plt.text(i, v + 0.5, str(v), ha='center')
            
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        return plt.gcf()

    except Exception as e:
        print(f"Erreur détaillée lors de la création du graphique: {str(e)}")
        print(f"Colonnes disponibles dans identifications: {identifications.columns.tolist()}")
        print(f"Colonnes disponibles dans feature_info: {feature_info.columns.tolist()}")
        raise

def plot_level1_molecules_per_sample(output_dir: Union[str, Path]) -> plt.Figure:
    """
    Graphique des molécules niveau 1 uniques par échantillon.
    """
    try:
        # Lire les fichiers
        identifications = pd.read_parquet(Path(output_dir) / "feature_matrix/feature_identifications.parquet")
        feature_info = pd.read_parquet(Path(output_dir) / "feature_matrix/feature_info.parquet")
        
        # Obtenir les comptes pour le niveau 1
        molecule_counts = get_molecules_per_sample(identifications, feature_info, confidence_level=1)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=molecule_counts, x='sample', y='n_molecules', color='green')
        plt.title("Nombre de molécules niveau 1 par échantillon")
        plt.xlabel("Échantillon")
        plt.ylabel("Nombre de molécules niveau 1")
        
        # Ajouter les valeurs sur les barres
        for i, v in enumerate(molecule_counts['n_molecules']):
            plt.text(i, v + 0.5, str(v), ha='center')
            
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        return plt.gcf()

    except Exception as e:
        print(f"Erreur détaillée lors de la création du graphique: {str(e)}")
        print(f"Colonnes disponibles dans identifications: {identifications.columns.tolist()}")
        print(f"Colonnes disponibles dans feature_info: {feature_info.columns.tolist()}")
        raise

def create_similarity_matrix(matches_df: pd.DataFrame, feature_info: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Crée une matrice de similarité basée sur les molécules uniques communes.
    """
    # Créer un DataFrame avec une ligne par couple molécule-échantillon
    all_sample_molecules = []
    for _, match in matches_df.iterrows():
        feature_samples = feature_info.loc[match['feature_idx'], 'samples'].split(',')
        for sample in feature_samples:
            all_sample_molecules.append({
                'sample': sample,
                'molecule': match['match_name']
            })
    
    df_expanded = pd.DataFrame(all_sample_molecules)
    
    # Créer une matrice binaire échantillons x molécules
    molecule_matrix = pd.crosstab(
        df_expanded['sample'],
        df_expanded['molecule']
    ).astype(float)
    
    # Calculer la matrice de similarité de Jaccard
    similarity_matrix = pd.DataFrame(
        0.0,
        index=molecule_matrix.index,
        columns=molecule_matrix.index,
        dtype=float
    )
    
    for idx1 in molecule_matrix.index:
        for idx2 in molecule_matrix.index:
            vec1 = molecule_matrix.loc[idx1] > 0
            vec2 = molecule_matrix.loc[idx2] > 0
            intersection = (vec1 & vec2).sum()
            union = (vec1 | vec2).sum()
            if union > 0:
                similarity = (intersection / union) * 100
                similarity_matrix.loc[idx1, idx2] = float(similarity)
    
    return similarity_matrix, molecule_matrix

def plot_sample_similarity_heatmap(output_dir: Union[str, Path]) -> plt.Figure:
    """Génère une heatmap de similarité entre échantillons."""
    try:
        matches_df = pd.read_parquet(Path(output_dir) / "feature_matrix/feature_identifications.parquet")
        feature_info = pd.read_parquet(Path(output_dir) / "feature_matrix/feature_info.parquet")
        
        similarity_matrix, _ = create_similarity_matrix(matches_df, feature_info)
        
        similarity_array = similarity_matrix.to_numpy(dtype=float)
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
        
        plt.title("Similarité entre échantillons (%)")
        return g.figure

    except Exception as e:
        print(f"Erreur lors de la création de la heatmap: {str(e)}")
        raise

def plot_sample_similarity_heatmap_by_confidence(output_dir: Union[str, Path], 
                                               confidence_levels: List[int],
                                               title_suffix: str = "") -> plt.Figure:
    """Génère une heatmap de similarité pour des niveaux de confiance spécifiques."""
    try:
        matches_df = pd.read_parquet(Path(output_dir) / "feature_matrix/feature_identifications.parquet")
        feature_info = pd.read_parquet(Path(output_dir) / "feature_matrix/feature_info.parquet")
        
        # Filtrer par niveaux de confiance
        filtered_df = matches_df[matches_df['confidence_level'].isin(confidence_levels)]
        
        similarity_matrix, _ = create_similarity_matrix(filtered_df, feature_info)
        
        similarity_array = similarity_matrix.to_numpy(dtype=float)
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

    except Exception as e:
        print(f"Erreur lors de la création de la heatmap: {str(e)}")
        raise

def plot_level1_molecule_distribution_bubble(output_dir: Union[str, Path], top_n: int = 20) -> plt.Figure:
    """
    Crée un bubble plot pour les molécules de niveau 1 avec les intensités spécifiques à chaque échantillon.
    """
    try:
        # Lire les fichiers
        identifications = pd.read_parquet(Path(output_dir) / "feature_matrix/feature_identifications.parquet")
        feature_info = pd.read_parquet(Path(output_dir) / "feature_matrix/feature_info.parquet")
        feature_matrix = pd.read_parquet(Path(output_dir) / "feature_matrix/feature_matrix.parquet")
        
        # Filtrer niveau 1
        level1_df = identifications[identifications['confidence_level'] == 1].copy()
        
        intensities_data = []
        
        # Parcourir chaque molécule unique
        for molecule in level1_df['match_name'].unique():
            molecule_matches = level1_df[level1_df['match_name'] == molecule]
            
            # Pour chaque match
            for _, match in molecule_matches.iterrows():
                feature_idx = match['feature_idx']
                feature_info_row = feature_info.loc[feature_idx]
                feature_id = f"{feature_info_row['feature_id']}_mz{feature_info_row['mz']:.4f}"
                
                if feature_id in feature_matrix.columns:
                    # Obtenir les intensités spécifiques à chaque échantillon
                    for sample in feature_matrix.index:
                        intensity = feature_matrix.loc[sample, feature_id]
                        if intensity > 0:
                            intensities_data.append({
                                'molecule': molecule,
                                'sample': sample,
                                'intensity': intensity,
                                'adduct': match['match_adduct']
                            })
        
        # Créer le DataFrame et garder la plus forte intensité
        intensity_df = pd.DataFrame(intensities_data)
        if intensity_df.empty:
            raise ValueError("Aucune donnée d'intensité trouvée")
            
        pivot_df = (intensity_df.groupby(['molecule', 'sample'])['intensity']
                   .max()
                   .unstack(fill_value=0))
        
        # Sélectionner les top_n molécules les plus fréquentes
        molecule_presence = (pivot_df > 0).sum()
        top_molecules = molecule_presence.nlargest(top_n).index
        pivot_df = pivot_df[top_molecules]
        
        # Trouver l'intensité maximale globale pour normaliser
        global_max_intensity = pivot_df.max().max()
        
        # Créer le bubble plot
        plt.figure(figsize=(20, 10))
        
        # Pour chaque molécule
        for molecule_idx, molecule in enumerate(pivot_df.columns):
            intensities = pivot_df[molecule]
            
            if intensities.max() > 0:
                # Normaliser les tailles par rapport à l'intensité maximale globale
                sizes = (intensities / global_max_intensity * 1000).values
                colors = intensities.values  # Garder les vraies intensités pour les couleurs
                
                plt.scatter([molecule_idx] * len(pivot_df.index), 
                          range(len(pivot_df.index)),
                          s=sizes,
                          c=colors,
                          cmap='viridis',
                          alpha=0.6,
                          vmin=0,
                          vmax=global_max_intensity)  # Fixer l'échelle de couleur
                
                # Ajouter les valeurs d'intensité
                for sample_idx, intensity in enumerate(intensities):
                    if intensity > 0:
                        plt.annotate(f'{int(intensity)}',
                                   (molecule_idx, sample_idx),
                                   xytext=(5, 5),
                                   textcoords='offset points',
                                   fontsize=8)
        
        plt.xticks(range(len(top_molecules)), top_molecules, rotation=45, ha='right')
        plt.yticks(range(len(pivot_df.index)), pivot_df.index)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        cbar = plt.colorbar(label='Intensité')
        cbar.formatter.set_scientific(False)
        cbar.update_ticks()
        
        plt.title('Distribution des molécules de niveau 1 par échantillon')
        plt.tight_layout()
        
        return plt.gcf()

    except Exception as e:
        print(f"Erreur lors de la création du bubble plot: {str(e)}")
        raise

def plot_tics_interactive(input_dir: Union[str, Path], output_dir: Union[str, Path]) -> None:
    """Crée un plot interactif des TIC MS1."""
    try:
        tics = {}
        for file_path in Path(input_dir).glob("*.parquet"):
            data = pd.read_parquet(file_path)
            tic = (data[data['mslevel'] == "1"]
                  .groupby(['rt', 'scanid'])['intensity']
                  .sum()
                  .reset_index()
                  .sort_values('rt'))
            tics[file_path.stem] = tic
        
        if not tics:
            print("Aucun TIC calculé")
            return
        
        fig = go.Figure()
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
        
        fig.update_layout(
            title="TIC MS1",
            xaxis_title="Temps de rétention (min)",
            yaxis_title="Intensité",
            template='plotly_white',
            width=1200,
            height=600,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.write_html(Path(output_dir) / "tic_ms1_comparison.html")
        fig.write_image(Path(output_dir) / "tic_ms1_comparison.png")

    except Exception as e:
        print(f"Erreur lors de la création du TIC: {str(e)}")
        raise

def analyze_sample_clusters(input_dir: Union[str, Path], n_clusters: int = 3) -> Dict:
    """
    Analyse les clusters d'échantillons basés sur leurs profils moléculaires.
    """
    try:
        # Charger les données
        identifications = pd.read_parquet(Path(input_dir) / "feature_matrix/feature_identifications.parquet")
        feature_info = pd.read_parquet(Path(input_dir) / "feature_matrix/feature_info.parquet")
        
        # Créer un DataFrame avec une ligne par couple molécule-échantillon
        all_sample_molecules = []
        for _, match in identifications.iterrows():
            feature_samples = feature_info.loc[match['feature_idx'], 'samples'].split(',')
            for sample in feature_samples:
                all_sample_molecules.append({
                    'sample': sample,
                    'molecule': match['match_name'],
                    'intensity': match['peak_intensity']
                })
        
        df_expanded = pd.DataFrame(all_sample_molecules)
        
        # Créer la matrice pivot avec les intensités maximales
        pivot_df = df_expanded.pivot_table(
            index='sample',
            columns='molecule',
            values='intensity',
            aggfunc='max'
        ).fillna(0)
        
        # Ajuster le nombre de clusters en fonction du nombre d'échantillons
        n_samples = len(pivot_df)
        actual_n_clusters = min(n_clusters, n_samples)
        
        if actual_n_clusters < 2:
            # Si nous avons moins de 2 échantillons, retourner des statistiques simples
            cluster_stats = {
                'Groupe 1': {
                    'samples': list(pivot_df.index),
                    'n_samples': n_samples,
                    'avg_molecules_per_sample': (pivot_df > 0).sum(axis=1).mean(),
                    'characteristic_molecules': pivot_df.mean().sort_values(ascending=False).index.tolist()
                }
            }
            return cluster_stats
        
        # Normaliser les données
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(pivot_df)
        
        # Appliquer K-means
        kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Analyser chaque cluster
        cluster_stats = {}
        for i in range(actual_n_clusters):
            cluster_samples = pivot_df.index[clusters == i]
            cluster_data = pivot_df.loc[cluster_samples]
            
            # Identifier les molécules caractéristiques
            mean_intensities = cluster_data.mean()
            other_clusters_mean = pivot_df.loc[~pivot_df.index.isin(cluster_samples)].mean()
            fold_change = mean_intensities / (other_clusters_mean + 1e-10)  # Éviter division par zéro
            characteristic_molecules = fold_change.sort_values(ascending=False).index.tolist()
            
            cluster_stats[f'Cluster {i+1}'] = {
                'samples': list(cluster_samples),
                'n_samples': len(cluster_samples),
                'avg_molecules_per_sample': (cluster_data > 0).sum(axis=1).mean(),
                'characteristic_molecules': characteristic_molecules
            }
        
        return cluster_stats
        
    except Exception as e:
        print(f"Erreur lors de l'analyse des clusters: {str(e)}")
        raise

def plot_cluster_statistics(cluster_stats: Dict) -> plt.Figure:
    """
    Crée une visualisation des statistiques des clusters.
    """
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extraire les données
        clusters = list(cluster_stats.keys())
        n_samples = [stats['n_samples'] for stats in cluster_stats.values()]
        avg_molecules = [stats['avg_molecules_per_sample'] for stats in cluster_stats.values()]
        
        # Graphique du nombre d'échantillons par cluster
        bars1 = ax1.bar(clusters, n_samples, color='skyblue')
        ax1.set_title("Nombre d'échantillons par cluster")
        ax1.set_ylabel("Nombre d'échantillons")
        ax1.tick_params(axis='x', rotation=45)
        
        # Ajouter les valeurs sur les barres
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        # Graphique de la moyenne de molécules par échantillon
        bars2 = ax2.bar(clusters, avg_molecules, color='lightgreen')
        ax2.set_title("Moyenne de molécules par échantillon")
        ax2.set_ylabel("Nombre moyen de molécules")
        ax2.tick_params(axis='x', rotation=45)
        
        # Ajouter les valeurs sur les barres
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        return plt.gcf()
        
    except Exception as e:
        print(f"Erreur lors de la création du graphique des clusters: {str(e)}")
        raise

def analyze_and_save_clusters(output_dir: Path) -> None:
    """
    Analyse et sauvegarde les statistiques des clusters.
    """
    try:
        # Le nombre de clusters sera automatiquement ajusté dans analyze_sample_clusters
        cluster_stats = analyze_sample_clusters(output_dir, n_clusters=3)
        
        # Sauvegarder l'analyse textuelle
        with open(output_dir / "cluster_analysis.txt", "w", encoding='utf-8') as f:
            f.write("Analyse des clusters d'échantillons\n")
            f.write("================================\n\n")
            
            # Statistiques globales
            total_samples = sum(stats['n_samples'] for stats in cluster_stats.values())
            avg_molecules_global = np.mean([stats['avg_molecules_per_sample'] 
                                          for stats in cluster_stats.values()])
            
            f.write(f"Statistiques globales:\n")
            f.write(f"- Nombre total d'échantillons: {total_samples}\n")
            f.write(f"- Moyenne globale de molécules par échantillon: {avg_molecules_global:.1f}\n\n")
            
            # Détails par cluster
            for cluster_name, stats in cluster_stats.items():
                f.write(f"\n{cluster_name}:\n")
                f.write(f"Nombre d'échantillons: {stats['n_samples']}\n")
                f.write(f"Moyenne de molécules par échantillon: {stats['avg_molecules_per_sample']:.1f}\n")
                
                f.write("\nMolécules caractéristiques:\n")
                for idx, molecule in enumerate(stats['characteristic_molecules'][:10], 1):
                    f.write(f"{idx}. {molecule}\n")
                
                f.write("\nÉchantillons dans ce cluster:\n")
                for sample in sorted(stats['samples']):
                    f.write(f"- {sample}\n")
                f.write("\n" + "-"*50 + "\n")

        # Créer et sauvegarder les visualisations
        fig_stats = plot_cluster_statistics(cluster_stats)
        fig_stats.savefig(output_dir / "cluster_statistics.png", bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"   ✓ Analyse des clusters sauvegardée dans {output_dir}")
        
    except Exception as e:
        print(f"Erreur lors de l'analyse des clusters: {str(e)}")
        raise
