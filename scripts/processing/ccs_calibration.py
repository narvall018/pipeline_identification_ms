# scripts/processing/ccs_calibration.py
import pandas as pd
import numpy as np
import deimos
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class CCSCalibrator:
    def __init__(self, calibration_file):
        """
        Initialise le calibrateur CCS.
        
        Args:
            calibration_file: Chemin vers le fichier de calibration CCS
        """
        self.calibration_data = self._load_calibration_data(calibration_file)
        self.calibration_model = None
        
    def _load_calibration_data(self, file_path):
        """Charge et prépare les données de calibration"""
        try:
            df = pd.read_csv(file_path)
            
            # Calcul CCS si nécessaire (si seulement rCCS est présent)
            if 'CCS' not in df.columns and 'Reference rCCS' in df.columns:
                logger.info("Calcul des CCS à partir des rCCS")
                df['Mi'] = df['Reference m/z'] * df['z']
                df['mu'] = (df['Mi'] * 28.013) / (df['Mi'] + 28.013)
                df['sqrt_mu'] = np.sqrt(df['mu'])
                df['CCS'] = df['Reference rCCS'] * np.sqrt(1/df['mu']) * df["z"]
            
            logger.info("Données de calibration chargées avec succès")
            return df
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données de calibration: {str(e)}")
            raise
            
    def calibrate(self):
        """Effectue la calibration"""
        try:
            logger.info("Début de la calibration CCS")
            
            self.calibration_model = deimos.calibration.calibrate_ccs(
                mz=self.calibration_data['Measured m/z'],
                ta=self.calibration_data['Measured Time'],
                ccs=self.calibration_data['CCS'],
                q=self.calibration_data['z'],
                buffer_mass=28.013,  # Masse de l'azote
                power=True
            )
            
            logger.info("Calibration CCS terminée avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de la calibration: {str(e)}")
            raise
            
    def calculate_ccs(self, peaks_df):
        """
        Calcule les valeurs CCS pour un DataFrame de pics.
        
        Args:
            peaks_df: DataFrame contenant les pics (avec mz et drift_time)
            
        Returns:
            DataFrame avec les valeurs CCS ajoutées
        """
        try:
            if self.calibration_model is None:
                self.calibrate()
            
            df = peaks_df.copy()
            
            # Récupération de la charge depuis les données de calibration
            default_charge = self.calibration_data['z'].iloc[0]
            
            # Calcul CCS pour chaque pic
            df['CCS'] = df.apply(
                lambda row: self.calibration_model.arrival2ccs(
                    mz=row['mz'],
                    ta=row['drift_time'],
                    q=default_charge
                ),
                axis=1
            )
            
            logger.info(f"CCS calculées pour {len(df)} pics")
            return df
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul des CCS: {str(e)}")
            raise
