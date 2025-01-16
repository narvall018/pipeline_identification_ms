# scripts/visualization/plot_concentrations.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

# Supprimer les warnings
warnings.filterwarnings('ignore')

# Configure matplotlib to use 'Agg' backend
plt.switch_backend('Agg')

def plot_top_concentrations(quant_dir: Path, output_dir: Path, n_compounds: int = 5) -> None:
    """
    Plot les n molécules les plus concentrées pour chaque échantillon.
    
    Args:
        quant_dir: Dossier contenant les fichiers de quantification
        output_dir: Dossier de sortie pour les plots
        n_compounds: Nombre de composés à afficher (par défaut: 5)
    """
    print("\n📊 Génération des plots de concentration...")
    
    # Créer le dossier de sortie
    plots_dir = output_dir / "concentration_plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Trouver tous les fichiers de quantification
    quantification_files = list(quant_dir.glob("*_quantification.csv"))
    if not quantification_files:
        print("   ℹ️ Aucun fichier de quantification trouvé")
        return
        
    print(f"   ✓ {len(quantification_files)} fichiers de quantification trouvés")
    
    # Configurer le style seaborn
    sns.set_style("whitegrid")
    
    # Traiter chaque fichier
    for file in quantification_files:
        try:
            sample_name = file.stem.replace("_quantification", "")
            print(f"      • Traitement de {sample_name}")
            
            # Lire et préparer les données
            df = pd.read_csv(file)
            df["conc"] = df['conc'] * 1e6
            top_compounds = df.nlargest(n_compounds, 'conc')
            
            # Créer le plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Mise à jour de l'appel à barplot pour éviter le warning
            sns.barplot(data=top_compounds, 
                       y='identifier', 
                       x='conc',
                       hue='identifier',
                       palette='viridis',
                       legend=False,
                       ax=ax)
            
            ax.set_title(f"Top {n_compounds} composés les plus concentrés\n{sample_name}")
            ax.set_xlabel("Concentration (ug/L)")
            ax.set_ylabel("Composé")
            
            # Ajuster la mise en page
            plt.tight_layout()
            
            # Sauvegarder
            output_file = plots_dir / f"{sample_name}_top_concentrations.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            print(f"      ❌ Erreur: {str(e)}")
            continue
    
    print(f"   ✓ Plots sauvegardés dans: {plots_dir}")
