# scripts/processing/identification.py
import pandas as pd
import h5py
from pathlib import Path
import logging
from ..config.config import Config
from ..utils.matching_utils import find_matches_window, calculate_match_scores

logger = logging.getLogger(__name__)

class CompoundIdentifier:
    def __init__(self):
        self.load_database()
        
    def load_database(self):
        """Charge la base de données"""
        try:
            db_path = Path(Config.INPUT_DATABASES) / Config.IDENTIFICATION['database_file']
            self.db = pd.read_hdf(db_path, key=Config.IDENTIFICATION['database_key'])
            logger.info(f"Base de données chargée: {len(self.db)} composés")
        except Exception as e:
            logger.error(f"Erreur chargement base de données: {str(e)}")
            raise
            
    def identify_compounds(self, peaks_df, output_dir):
        """Identifie les composés pour un ensemble de pics"""
        logger.info("Début de l'identification des composés")
        
        # Trouver tous les matches possibles
        matches_df = find_matches_window(peaks_df, self.db)
        
        if matches_df.empty:
            logger.warning("Aucun match trouvé")
            return None
        
        # Créer le dossier de sortie
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder tous les matches
        matches_path = output_dir / 'all_matches.parquet'
        matches_df.to_parquet(matches_path)
        
        logger.info(f"Matches sauvegardés: {matches_path}")
        return matches_df
