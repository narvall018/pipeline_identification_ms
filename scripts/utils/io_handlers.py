# scripts/utils/io_handlers.py
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import logging
import re

logger = logging.getLogger(__name__)

def sanitize_filename(filename):
    """
    Nettoie le nom de fichier en remplaçant les caractères problématiques
    """
    # Remplace les espaces par des underscores
    filename = filename.replace(' ', '_')
    
    # Remplace tout caractère non alphanumérique (sauf underscore et point) par un underscore
    filename = re.sub(r'[^\w.-]', '_', filename)
    
    return filename

def read_parquet_data(file_path):
    """Lecture des fichiers parquet"""
    try:
        parquet_file = pq.ParquetFile(file_path)
        table = parquet_file.read()
        data = table.to_pandas()
        metadata = table.schema.metadata
        logger.info(f"Fichier lu avec succès: {file_path}")
        return data, metadata
    except Exception as e:
        logger.error(f"Erreur lecture fichier {file_path}: {str(e)}")
        raise

def save_peaks(df, sample_name, step, data_type='samples', metadata=None):
    """
    Sauvegarde les résultats de la détection de pics
    """
    try:
        # Nettoie le nom de l'échantillon
        safe_sample_name = sanitize_filename(sample_name)
        logger.info(f"Nom d'échantillon nettoyé: {safe_sample_name}")
        
        # Création du chemin
        base_dir = Path(f"data/intermediate/{data_type}/{safe_sample_name}/ms1")
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Chemin complet du fichier
        file_path = base_dir / f"{step}.parquet"
        
        # Conversion en table Arrow
        table = pa.Table.from_pandas(df, preserve_index=False)
        if metadata:
            table = table.replace_schema_metadata(metadata)
        
        # Conversion en string du chemin et sauvegarde
        pq.write_table(table, str(file_path))
        logger.info(f"Données sauvegardées: {file_path}")
        
        return file_path
    
    except Exception as e:
        logger.error(f"Erreur sauvegarde {step} pour {sample_name}: {str(e)}")
        raise
