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

def get_molecules_per_sample(merged_df: pd.DataFrame, confidence_level: int = None) -> pd.DataFrame:
    """
    Compte les molécules uniques par échantillon.
    """
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
    """Génère un graphique du nombre de molécules uniques par échantillon."""
    try:
        # Lire le fichier features_complete
        merged_df = pd.read_parquet(Path(output_dir) / "feature_matrix" / "features_complete.parquet")
        
        # Obtenir les comptes
        molecule_counts = get_molecules_per_sample(merged_df)
        
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
        raise

def plot_level1_molecules_per_sample(output_dir: Union[str, Path]) -> plt.Figure:
    """Graphique des molécules niveau 1 uniques par échantillon."""
    try:
        # Lire le fichier features_complete
        merged_df = pd.read_parquet(Path(output_dir) / "feature_matrix" / "features_complete.parquet")
        
        # Obtenir les comptes pour le niveau 1
        molecule_counts = get_molecules_per_sample(merged_df, confidence_level=1)
        
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
        raise

def create_similarity_matrix(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée une matrice de similarité basée sur les molécules uniques communes.
    """
    # Créer un DataFrame avec une ligne par couple molécule-échantillon
    all_sample_molecules = []
    for _, match in merged_df.iterrows():
        samples = match['samples'].split(',')
        for sample in samples:
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
    
    return similarity_matrix

def plot_sample_similarity_heatmap(output_dir: Union[str, Path]) -> plt.Figure:
    """Génère une heatmap de similarité entre échantillons."""
    try:
        merged_df = pd.read_parquet(Path(output_dir) / "feature_matrix" / "features_complete.parquet")
        similarity_matrix = create_similarity_matrix(merged_df)
        
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
        merged_df = pd.read_parquet(Path(output_dir) / "feature_matrix" / "features_complete.parquet")
        
        # Filtrer par niveaux de confiance
        filtered_df = merged_df[merged_df['confidence_level'].isin(confidence_levels)]
        
        similarity_matrix = create_similarity_matrix(filtered_df)
        
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
        # Charger les données
        feature_matrix = pd.read_parquet(Path(output_dir) / "feature_matrix" / "feature_matrix.parquet")
        merged_df = pd.read_parquet(Path(output_dir) / "feature_matrix" / "features_complete.parquet")
        
        # Filtrer niveau 1
        level1_df = merged_df[merged_df['confidence_level'] == 1].copy()
        
        intensities_data = []
        
        # Pour chaque molécule de niveau 1
        for _, row in level1_df.iterrows():
            feature_id = f"{row['feature_id']}_mz{row['mz']:.4f}"
            
            # Récupérer les intensités de la matrice pour cette feature
            if feature_id in feature_matrix.columns:
                for sample in feature_matrix.index:
                    intensity = feature_matrix.loc[sample, feature_id]
                    if intensity > 0:  # Ne garder que les intensités non nulles
                        intensities_data.append({
                            'molecule': row['match_name'],
                            'sample': sample,
                            'intensity': intensity
                        })
        
        # Créer le DataFrame et garder la plus forte intensité
        intensity_df = pd.DataFrame(intensities_data)
        if intensity_df.empty:
            raise ValueError("Aucune donnée d'intensité trouvée")
            
        pivot_df = (intensity_df.groupby(['molecule', 'sample'])['intensity']
                   .max()
                   .unstack(fill_value=0))
        
        # Sélectionner les top_n molécules les plus fréquentes
        molecule_presence = (pivot_df > 0).sum(axis=1)
        top_molecules = molecule_presence.nlargest(top_n).index
        pivot_df = pivot_df.loc[top_molecules]
        
        # Trouver l'intensité maximale globale pour normaliser
        global_max_intensity = pivot_df.max().max()
        
        # Créer le bubble plot
        plt.figure(figsize=(20, 10))
        
        # Pour chaque molécule
        for molecule_idx, molecule in enumerate(pivot_df.index):
            intensities = pivot_df.loc[molecule]
            
            if intensities.max() > 0:
                # Normaliser les tailles par rapport à l'intensité maximale globale
                sizes = (intensities / global_max_intensity * 1000).values
                colors = intensities.values  # Garder les vraies intensités pour les couleurs
                
                plt.scatter([molecule_idx] * len(pivot_df.columns), 
                          range(len(pivot_df.columns)),
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
        plt.yticks(range(len(pivot_df.columns)), pivot_df.columns)
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

def analyze_sample_clusters(merged_df: pd.DataFrame, n_clusters: int = 3) -> Dict:
    """
    Analyse les clusters d'échantillons basés sur leurs profils moléculaires.
    
    Args:
        merged_df: DataFrame contenant les données features_complete
        n_clusters: nombre de clusters souhaité
    """
    try:
        # Créer un DataFrame avec une ligne par couple molécule-échantillon
        all_sample_molecules = []
        for _, match in merged_df.iterrows():
            samples = match['samples'].split(',')
            for sample in samples:
                all_sample_molecules.append({
                    'sample': sample,
                    'molecule': match['match_name'],
                    'intensity': match['intensity']
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

def analyze_and_save_clusters(output_dir: Path) -> None:
    """
    Analyse et sauvegarde les statistiques des clusters.
    """
    try:
        # Charger les données
        features_file = output_dir / "feature_matrix" / "features_complete.parquet"
        merged_df = pd.read_parquet(features_file)
        
        # Le nombre de clusters sera automatiquement ajusté dans analyze_sample_clusters
        cluster_stats = analyze_sample_clusters(merged_df, n_clusters=3)
        
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


def plot_categories_distribution_by_level(output_dir: Union[str, Path], 
                                        confidence_levels: List[int],
                                        title_suffix: str = "") -> plt.Figure:
    """
    Génère un graphique montrant la distribution des catégories par échantillon pour des niveaux de confiance spécifiques.
    """
    try:
        merged_df = pd.read_parquet(Path(output_dir) / "feature_matrix" / "features_complete.parquet")
        
        categories_data = []
        for _, row in merged_df.iterrows():
            if row['confidence_level'] not in confidence_levels:
                continue
                
            try:
                # Extraire les catégories
                if isinstance(row['categories'], str):
                    cats = eval(row['categories'])
                elif isinstance(row['categories'], list):
                    cats = row['categories']
                elif hasattr(row['categories'], 'tolist'):
                    cats = row['categories'].tolist()
                else:
                    continue
                
                # Pour chaque échantillon
                for sample in row['samples'].split(','):
                    if isinstance(cats, list):
                        for category in cats:
                            categories_data.append({
                                'sample': sample,
                                'category': category,
                                'confidence_level': row['confidence_level']
                            })
            except (ValueError, SyntaxError):
                continue
        
        if not categories_data:
            raise ValueError(f"Aucune donnée de catégorie trouvée pour les niveaux {confidence_levels}")
            
        cat_df = pd.DataFrame(categories_data)
        
        # Créer le graphique
        plt.figure(figsize=(15, 8))
        
        samples = sorted(cat_df['sample'].unique())
        categories = sorted(cat_df['category'].unique())
        
        x_positions = np.arange(len(samples))
        width = 0.8 / len(categories)  # Ajuster la largeur en fonction du nombre de catégories
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        
        # Pour chaque catégorie
        for i, category in enumerate(categories):
            heights = []
            for sample in samples:
                count = len(cat_df[(cat_df['sample'] == sample) & 
                                 (cat_df['category'] == category)])
                heights.append(count)
            
            x = x_positions + i * width
            plt.bar(x, heights, width, label=category, color=colors[i], alpha=0.8)
        
        plt.xlabel('Échantillons')
        plt.ylabel('Nombre de molécules')
        levels_text = f"niveau{'s' if len(confidence_levels) > 1 else ''} {', '.join(map(str, confidence_levels))}"
        plt.title(f'Distribution des catégories par échantillon - {levels_text}{title_suffix}')
        plt.xticks(x_positions + width * len(categories)/2, samples, rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        return plt.gcf()
        
    except Exception as e:
        print(f"Erreur lors de la création du graphique des catégories: {str(e)}")
        raise

def plot_categories_pie_charts(output_dir: Union[str, Path]) -> None:
    """
    Génère des diagrammes circulaires pour montrer la distribution des catégories
    pour différents niveaux de confiance.
    """
    try:
        merged_df = pd.read_parquet(Path(output_dir) / "feature_matrix" / "features_complete.parquet")
        
        # Créer trois subplots pour les différentes combinaisons de niveaux
        fig, axs = plt.subplots(1, 3, figsize=(20, 7))
        
        level_combinations = [
            ([1], "Niveau 1"),
            ([1, 2], "Niveaux 1-2"),
            ([1, 2, 3], "Niveaux 1-2-3")
        ]
        
        for idx, (levels, title) in enumerate(level_combinations):
            categories_count = {}
            filtered_df = merged_df[merged_df['confidence_level'].isin(levels)]
            
            for _, row in filtered_df.iterrows():
                try:
                    if isinstance(row['categories'], str):
                        cats = eval(row['categories'])
                    elif isinstance(row['categories'], list):
                        cats = row['categories']
                    elif hasattr(row['categories'], 'tolist'):
                        cats = row['categories'].tolist()
                    else:
                        continue
                        
                    if isinstance(cats, list):
                        for cat in cats:
                            categories_count[cat] = categories_count.get(cat, 0) + 1
                except (ValueError, SyntaxError):
                    continue
            
            if categories_count:
                # Trier par valeur décroissante
                categories_count = dict(sorted(categories_count.items(), key=lambda x: x[1], reverse=True))
                
                # Créer le camembert
                wedges, texts, autotexts = axs[idx].pie(
                    categories_count.values(),
                    labels=categories_count.keys(),
                    autopct='%1.1f%%',
                    colors=plt.cm.Set3(np.linspace(0, 1, len(categories_count))),
                    textprops={'fontsize': 8}
                )
                
                # Ajuster la taille des labels
                plt.setp(texts, size=8)
                plt.setp(autotexts, size=8)
                
                axs[idx].set_title(title)
            else:
                axs[idx].text(0.5, 0.5, 'Aucune donnée', 
                            ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig(output_dir / "categories_pie_charts.png", bbox_inches='tight', dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"Erreur lors de la création des camemberts: {str(e)}")
        raise

def analyze_categories(output_dir: Path) -> None:
    """
    Analyse et sauvegarde toutes les visualisations des catégories.
    """
    try:
        # Générer les distributions pour différents niveaux
        fig1 = plot_categories_distribution_by_level(output_dir, [1], " - Niveau 1")
        fig1.savefig(output_dir / "categories_distribution_level1.png", bbox_inches='tight', dpi=300)
        plt.close(fig1)

        fig2 = plot_categories_distribution_by_level(output_dir, [1, 2], " - Niveaux 1-2")
        fig2.savefig(output_dir / "categories_distribution_level12.png", bbox_inches='tight', dpi=300)
        plt.close(fig2)

        fig3 = plot_categories_distribution_by_level(output_dir, [1, 2, 3], " - Niveaux 1-2-3")
        fig3.savefig(output_dir / "categories_distribution_level123.png", bbox_inches='tight', dpi=300)
        plt.close(fig3)

        # Générer les camemberts
        plot_categories_pie_charts(output_dir)
        
        print(f"   ✓ Visualisations des catégories sauvegardées dans {output_dir}")
        
    except Exception as e:
        print(f"Erreur lors de l'analyse des catégories: {str(e)}")
        raise