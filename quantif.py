# quantif.py

from pathlib import Path
import subprocess
from scripts.quantification.compound_recovery import get_compound_summary

def find_csv_file(directory: Path) -> Path:
    csv_files = list(directory.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Aucun fichier CSV trouvé dans {directory}")
    return csv_files[0]

def main():
    # Chemins
    compounds_dir = Path("data/input/calibrants/compounds")
    calibration_dir = Path("data/input/calibrants/samples")
    output_dir = Path("output/quantification")
    input_dir = Path("output")
    r_script_path = Path("scripts/quantification/ms2quant_analysis.R")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Étape 1: Préparation des données avec Python
        compounds_file = find_csv_file(compounds_dir)
        calibration_file = find_csv_file(calibration_dir)
        
        summary_df = get_compound_summary(
            input_dir=input_dir,
            compounds_file=compounds_file,
            calibration_file=calibration_file,
            min_samples=4
        )
        
        if not summary_df.empty:
            summary_df.to_csv(output_dir / "compounds_summary.csv", index=False)
            print(f"✅ Données préparées dans {output_dir}")
            
            # Étape 2: Analyse MS2Quant avec R
            try:
                subprocess.run(["Rscript", str(r_script_path)], check=True)
                print("✅ Analyse MS2Quant terminée")
            except subprocess.CalledProcessError:
                print("❌ Erreur lors de l'exécution du script R")
            
    except FileNotFoundError:
        pass

if __name__ == "__main__":
    main()