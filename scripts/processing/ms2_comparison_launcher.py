import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def run_ms2_comparison():
    """
    Lance le script R de comparaison MS2
    """
    try:
        # Chemin vers le script R
        r_script_path = Path("scripts/processing/ms2_comparison.R")
        
        print("\n📊 Lancement de la comparaison des spectres MS2...")
        
        # Exécution du script R
        result = subprocess.run(
            ["Rscript", str(r_script_path)],
            capture_output=True,
            text=True
        )
        
        # Affichage de la sortie R
        print(result.stdout)
        
        # Vérification des erreurs
        if result.returncode != 0:
            print("❌ Erreur dans l'exécution du script R:")
            print(result.stderr)
            raise Exception("Échec de la comparaison MS2")
            
        print("✅ Comparaison MS2 terminée avec succès")
        
    except Exception as e:
        logger.error(f"Erreur lors de la comparaison MS2: {str(e)}")
        raise
