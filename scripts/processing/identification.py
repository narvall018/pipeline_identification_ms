#scripts/processing/identification.py
#-*- coding:utf-8 -*-

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple
from ..config.config import Config
from ..utils.matching_utils import find_matches_window

class CompoundIdentifier:
    """
    Classe pour identifier les composés à partir des données de pics.
    """
    def __init__(self) -> None:
        """Initialise l'identificateur en chargeant la base de données."""
        self.logger = logging.getLogger(__name__)
        self.config = Config.IDENTIFICATION
        self.db: pd.DataFrame = pd.DataFrame()
        self.load_database()

    def load_database(self) -> None:
        """Charge la base de données depuis le fichier HDF5 configuré."""
        try:
            # Construction du chemin vers le fichier de base de données
            db_path = Path(Config.PATHS.INPUT_DATABASES) / self.config.database_file

            # Vérification de l'existence du fichier
            if not db_path.exists():
                raise FileNotFoundError(f"Base de données non trouvée: {db_path}")

            # Chargement de la base de données
            self.db = pd.read_hdf(db_path, key=self.config.database_key)

            # Préparation des colonnes MS2 si présentes
            if 'peaks_ms2_mz' in self.db.columns and 'peaks_ms2_intensities' in self.db.columns:
                # Conversion des strings en listes si nécessaire
                for col in ['peaks_ms2_mz', 'peaks_ms2_intensities']:
                    self.db[col] = self.db[col].apply(self._convert_peaks_string_to_list)

            self.logger.info(f"Base de données chargée avec succès : {len(self.db)} composés")

        except Exception as e:
            self.logger.error(f"Erreur lors du chargement de la base de données : {str(e)}")
            raise

    def _convert_peaks_string_to_list(self, peaks_str: str) -> list:
        """
        Convertit une chaîne de pics en liste.
        
        Args:
            peaks_str: Chaîne de caractères représentant les pics
            
        Returns:
            list: Liste des valeurs de pics
        """
        try:
            if pd.isna(peaks_str):
                return []
            if isinstance(peaks_str, list):
                return peaks_str
            if isinstance(peaks_str, str):
                # Nettoie et convertit la chaîne en liste
                peaks_str = peaks_str.strip('[]')
                if not peaks_str:
                    return []
                return [float(x) for x in peaks_str.split(',')]
            return []
        except Exception as e:
            self.logger.warning(f"Erreur de conversion des pics: {str(e)}")
            return []

    def prepare_database_query(self, peaks_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prépare la base de données pour la recherche.
        
        Args:
            peaks_df: DataFrame des pics à identifier
            
        Returns:
            pd.DataFrame: Base de données préparée
        """
        try:
            # Vérification des colonnes requises
            required_columns = set(Config.DB_COLUMNS.values())
            if not required_columns.issubset(self.db.columns):
                missing = required_columns - set(self.db.columns)
                raise ValueError(f"Colonnes manquantes dans la base de données: {missing}")

            # Tri de la base de données pour optimiser la recherche
            return self.db.sort_values('mz')

        except Exception as e:
            self.logger.error(f"Erreur lors de la préparation de la requête : {str(e)}")
            raise

    def identify_compounds(
        self,
        peaks_df: pd.DataFrame,
        output_dir: str
    ) -> Optional[pd.DataFrame]:
        """
        Identifie les composés correspondants pour un ensemble de pics.
        
        Args:
            peaks_df: DataFrame des pics à identifier
            output_dir: Répertoire de sortie
            
        Returns:
            Optional[pd.DataFrame]: DataFrame des correspondances trouvées
        """
        self.logger.info("Début du processus d'identification des composés.")
        
        try:
            # Préparation de la base de données
            db_query = self.prepare_database_query(peaks_df)
            
            # Recherche des correspondances
            self.logger.info("Recherche des correspondances...")
            matches_df = find_matches_window(
                peaks_df=peaks_df,
                db_df=db_query,
                tolerances=self.config.tolerances
            )

            # Vérification des résultats
            if matches_df.empty:
                self.logger.warning("Aucune correspondance trouvée.")
                return None

            # Création du répertoire de sortie
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Sauvegarde des résultats
            matches_path = output_path / 'all_matches.parquet'
            matches_df.to_parquet(matches_path)

            # Statistiques d'identification
            self.logger.info(self._get_identification_stats(matches_df))
            
            return matches_df

        except Exception as e:
            self.logger.error(f"Erreur lors de l'identification des composés : {str(e)}")
            raise

    def _get_identification_stats(self, matches_df: pd.DataFrame) -> str:
        """
        Génère un résumé des statistiques d'identification.
        
        Args:
            matches_df: DataFrame des correspondances
            
        Returns:
            str: Résumé des statistiques
        """
        stats = []
        total_matches = len(matches_df)
        unique_compounds = matches_df['match_name'].nunique()
        stats.append(f"Total des correspondances : {total_matches}")
        stats.append(f"Composés uniques : {unique_compounds}")
        
        if 'confidence_level' in matches_df.columns:
            for level in sorted(matches_df['confidence_level'].unique()):
                level_count = len(matches_df[matches_df['confidence_level'] == level])
                level_percent = (level_count / total_matches) * 100
                stats.append(f"Niveau {level}: {level_count} ({level_percent:.1f}%)")
        
        return "\n".join(stats)

    def get_identification_metrics(self, matches_df: pd.DataFrame) -> Dict:
        """
        Calcule les métriques d'identification.
        
        Args:
            matches_df: DataFrame des correspondances
            
        Returns:
            Dict: Métriques calculées
        """
        try:
            metrics = {
                'total_matches': len(matches_df),
                'unique_compounds': matches_df['match_name'].nunique(),
                'confidence_levels': {},
                'mass_error_stats': {},
                'rt_error_stats': {},
                'ccs_error_stats': {}
            }

            # Statistiques par niveau de confiance
            if 'confidence_level' in matches_df.columns:
                for level in sorted(matches_df['confidence_level'].unique()):
                    level_df = matches_df[matches_df['confidence_level'] == level]
                    metrics['confidence_levels'][f'level_{level}'] = {
                        'count': len(level_df),
                        'percent': (len(level_df) / len(matches_df)) * 100,
                        'unique_compounds': level_df['match_name'].nunique()
                    }

            # Statistiques d'erreurs
            if 'mz_error_ppm' in matches_df.columns:
                metrics['mass_error_stats'] = self._calculate_error_stats(
                    matches_df['mz_error_ppm']
                )
            
            if 'rt_error_min' in matches_df.columns:
                metrics['rt_error_stats'] = self._calculate_error_stats(
                    matches_df['rt_error_min']
                )
            
            if 'ccs_error_percent' in matches_df.columns:
                metrics['ccs_error_stats'] = self._calculate_error_stats(
                    matches_df['ccs_error_percent']
                )

            return metrics

        except Exception as e:
            self.logger.error(f"Erreur lors du calcul des métriques : {str(e)}")
            return {}

    def _calculate_error_stats(self, error_series: pd.Series) -> Dict:
        """
        Calcule les statistiques d'erreur.
        
        Args:
            error_series: Série des erreurs
            
        Returns:
            Dict: Statistiques calculées
        """
        return {
            'mean': float(error_series.mean()),
            'std': float(error_series.std()),
            'median': float(error_series.median()),
            'min': float(error_series.min()),
            'max': float(error_series.max()),
            'abs_mean': float(error_series.abs().mean())
        }