import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

def plot_unique_molecules_per_sample(samples_dir):
    """
    Plot le nombre de molécules uniques par échantillon
    
    Args:
        samples_dir: Chemin vers le dossier contenant les dossiers d'échantillons
    """
    # Liste pour stocker les résultats
    results = []
    
    # Parcourir les dossiers d'échantillons
    samples_path = Path(samples_dir)
    for sample_dir in samples_path.iterdir():
        if sample_dir.is_dir():
            matches_file = sample_dir / 'ms1' / 'identifications' / 'all_matches.parquet'
            if matches_file.exists():
                # Charger les matches
                matches_df = pd.read_parquet(matches_file)
                # Compter les molécules uniques
                n_unique = len(matches_df['match_name'].unique())
                results.append({
                    'sample': sample_dir.name,
                    'n_molecules': n_unique
                })
    
    # Créer DataFrame
    results_df = pd.DataFrame(results)
    
    # Créer le plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x='sample', y='n_molecules')
    
    plt.title("Nombre de molécules uniques par échantillon")
    plt.xlabel("Échantillon")
    plt.ylabel("Nombre de molécules uniques")
    
    # Rotation des labels
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return plt.gcf()
