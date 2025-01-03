#scripts/utils/io_handlers.py
#-*- coding:utf-8 -*-

import re
import logging
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Tuple, Optional, Dict, Union
from ..config.config import Config

class IOHandler:
    """Classe pour gérer les opérations d'entrée/sortie du pipeline."""
    
    def __init__(self):
        """Initialise le gestionnaire IO avec la configuration."""
        self.logger = logging.getLogger(__name__)
        self.config = Config.PATHS

    def sanitize_filename(self, filename: str) -> str:
        """
        Nettoie le nom de fichier en remplaçant les caractères problématiques.
        
        Args:
            filename: Nom du fichier à nettoyer
            
        Returns:
            str: Nom de fichier nettoyé
        """
        try:
            # Remplace les espaces par des underscores
            filename = filename.replace(' ', '_')
            
            # Remplace tous les caractères non alphanumériques
            filename = re.sub(r'[^\w.-]', '_', filename)
            
            return filename

        except Exception as e:
            self.logger.error(f"Erreur lors du nettoyage du nom de fichier : {str(e)}")
            raise

    def read_parquet_data(
        self,
        file_path: Union[str, Path]
    ) -> Tuple[pd.DataFrame, Optional[Dict[bytes, bytes]]]:
        """
        Lit un fichier Parquet et retourne les données et métadonnées.
        
        Args:
            file_path: Chemin vers le fichier Parquet
            
        Returns:
            Tuple[pd.DataFrame, Optional[Dict]]: Données et métadonnées
        """
        try:
            # Conversion en Path si nécessaire
            file_path = Path(file_path)
            
            # Vérification de l'existence du fichier
            if not file_path.exists():
                raise FileNotFoundError(f"Fichier non trouvé : {file_path}")
            
            # Ouvre le fichier Parquet
            parquet_file = pq.ParquetFile(file_path)
            
            # Lit les données
            table = parquet_file.read()
            
            # Conversion en DataFrame
            data = table.to_pandas()
            
            # Récupération des métadonnées
            metadata = table.schema.metadata
            
            self.logger.info(f"Fichier Parquet lu avec succès : {file_path}")
            
            return data, metadata

        except Exception as e:
            self.logger.error(f"Erreur lors de la lecture du fichier {file_path} : {str(e)}")
            raise

    def save_peaks(
        self,
        df: pd.DataFrame,
        sample_name: str,
        step: str,
        data_type: str = 'samples',
        metadata: Optional[Dict[bytes, bytes]] = None
    ) -> Path:
        """
        Sauvegarde les résultats de la détection de pics.
        
        Args:
            df: DataFrame à sauvegarder
            sample_name: Nom de l'échantillon
            step: Étape de traitement
            data_type: Type de données
            metadata: Métadonnées optionnelles
            
        Returns:
            Path: Chemin du fichier sauvegardé
        """
        try:
            # Nettoie le nom de l'échantillon
            safe_sample_name = self.sanitize_filename(sample_name)
            self.logger.info(f"Nom de l'échantillon nettoyé : {safe_sample_name}")
            
            # Détermine le répertoire de sortie
            if data_type == 'samples':
                base_dir = self.config.INTERMEDIATE_SAMPLES
            elif data_type == 'calibrants':
                base_dir = self.config.INTERMEDIATE_CALIBRANTS
            else:
                base_dir = self.config.INTERMEDIATE_DIR / data_type
            
            # Crée le répertoire pour l'échantillon
            output_dir = base_dir / safe_sample_name / "ms1"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Chemin du fichier de sortie
            file_path = output_dir / f"{step}.parquet"
            
            # Conversion en table Arrow
            table = pa.Table.from_pandas(df, preserve_index=False)
            
            # Ajout des métadonnées si présentes
            if metadata:
                table = table.replace_schema_metadata(metadata)
            
            # Sauvegarde
            pq.write_table(table, str(file_path))
            self.logger.info(f"Données sauvegardées avec succès : {file_path}")
            
            return file_path

        except Exception as e:
            self.logger.error(
                f"Erreur lors de la sauvegarde de l'étape {step} "
                f"pour l'échantillon {sample_name} : {str(e)}"
            )
            raise

    def ensure_directory(self, directory: Union[str, Path]) -> Path:
        """
        S'assure qu'un répertoire existe, le crée si nécessaire.
        
        Args:
            directory: Chemin du répertoire
            
        Returns:
            Path: Chemin du répertoire créé
        """
        try:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            return directory
        except Exception as e:
            self.logger.error(f"Erreur lors de la création du répertoire {directory} : {str(e)}")
            raise

    def save_results(
        self,
        df: pd.DataFrame,
        output_path: Union[str, Path],
        save_csv: bool = True
    ) -> None:
        """
        Sauvegarde les résultats dans plusieurs formats.
        
        Args:
            df: DataFrame à sauvegarder
            output_path: Chemin de sortie
            save_csv: Si True, sauvegarde aussi en CSV
        """
        try:
            output_path = Path(output_path)
            
            # Crée le répertoire parent si nécessaire
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Sauvegarde en Parquet
            df.to_parquet(output_path.with_suffix('.parquet'))
            
            # Sauvegarde optionnelle en CSV
            if save_csv:
                df.to_csv(output_path.with_suffix('.csv'), index=False)
                
            self.logger.info(f"Résultats sauvegardés avec succès : {output_path}")

        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde des résultats : {str(e)}")
            raise

    def list_input_files(
        self,
        pattern: str = "*.parquet",
        data_type: str = 'samples'
    ) -> list:
        """
        Liste les fichiers d'entrée correspondant à un motif.
        
        Args:
            pattern: Motif de recherche des fichiers
            data_type: Type de données à rechercher
            
        Returns:
            list: Liste des chemins des fichiers trouvés
        """
        try:
            if data_type == 'samples':
                search_dir = self.config.INPUT_SAMPLES
            elif data_type == 'blanks':
                search_dir = self.config.INPUT_BLANKS
            elif data_type == 'calibrants':
                search_dir = self.config.INPUT_CALIBRANTS
            else:
                search_dir = self.config.INPUT_DIR / data_type
            
            return list(search_dir.glob(pattern))

        except Exception as e:
            self.logger.error(f"Erreur lors de la recherche des fichiers : {str(e)}")
            return []