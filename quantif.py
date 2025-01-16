# scripts/quantification/quantif.py
from pathlib import Path
import subprocess
import sys
import os
import pandas as pd

# Ajout du dossier courant au PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))

from scripts.quantification.compound_recovery import get_compound_summary
from scripts.visualization.plot_concentrations import plot_top_concentrations
from scripts.quantification.calculate_rqmix import calculate_rqmix

def find_csv_file(directory: Path) -> Path:
    """Trouve le premier fichier CSV dans un dossier."""
    csv_files = list(directory.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Aucun fichier CSV trouvé dans {directory}")
    return csv_files[0]

def run_r_script() -> bool:
    """Lance le script R de MS2Quant."""
    r_script_path = Path("scripts/quantification/ms2quant_analysis.R")
    try:
        subprocess.run(["Rscript", str(r_script_path)], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de l'exécution du script R")
        return False

def main() -> None:
    """Point d'entrée principal pour la quantification."""
    # Définition des chemins
    compounds_dir = Path("data/input/calibrants/compounds")
    calibration_dir = Path("data/input/calibrants/samples")
    output_dir = Path("output/quantification")
    input_dir = Path("data/output")
    
    print("\n" + "="*80)
    print("QUANTIFICATION DES ÉCHANTILLONS")
    print("="*80)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print("\n🔍 Préparation des données...")
        compounds_file = find_csv_file(compounds_dir)
        calibration_file = find_csv_file(calibration_dir)
        
        summary_df = get_compound_summary(
            input_dir=input_dir,
            compounds_file=compounds_file,
            calibration_file=calibration_file
        )
        #summary_df = pd.read_csv('output/quantification/compounds_summary.csv')
        
        if not summary_df.empty:
            summary_df.to_csv(output_dir / "compounds_summary.csv", index=False)
            print(f"   ✓ Données préparées dans {output_dir}")
            
            print("\n🧪 Lancement de l'analyse MS2Quant...")
            if run_r_script():
                print("   ✓ Analyse MS2Quant terminée")
                
                print("\n📊 Calcul des RQmix...")
                calculate_rqmix(output_dir / "samples_quantification", output_dir)
                print("   ✓ Calcul RQmix terminé")
                
                print("\n📊 Génération des visualisations...")
                quant_dir = output_dir / "samples_quantification"
                plot_top_concentrations(quant_dir, output_dir)
                
                print("\n✅ Quantification terminée avec succès")
            
    except FileNotFoundError as e:
        print(f"\n❌ Erreur: {str(e)}")
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {str(e)}")

if __name__ == "__main__":
    main()
