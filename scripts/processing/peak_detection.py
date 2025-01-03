#scripts/processing/peak_detection.py
#-*- coding:utf-8 -*-

import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict
import deimos
from sklearn.cluster import DBSCAN
from ..config.config import Config

logger = logging.getLogger(__name__)

class PeakDetector:
    """Classe responsable de la détection et du clustering des pics MS1."""
    
    def __init__(self):
        """Initialise le détecteur de pics avec la configuration."""
        self.peak_config = Config.PEAK_DETECTION
        self.cluster_config = Config.INTRA_CLUSTERING
        self.logger = logging.getLogger(__name__)

    def prepare_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Prépare les données MS1 pour la détection de pics.
        
        Args:
            df (pd.DataFrame): Données brutes du fichier d'entrée
            
        Returns:
            Optional[pd.DataFrame]: Données préparées ou None si échec
        """
        try:
            df['mslevel'] = df['mslevel'].astype(int)
            data = df[df['mslevel'] == 1].copy()

            if len(data) == 0:
                self.logger.warning("Aucune donnée MS1 trouvée.")
                return None

            for col in ['mz', 'intensity', 'rt', 'dt']:
                data[col] = data[col].astype(float)

            data = data.rename(columns={
                'rt': 'retention_time',
                'dt': 'drift_time',
                'intensity': 'intensity',
                'scanid': 'scanId'
            })

            columns = ['mz', 'intensity', 'drift_time', 'retention_time']
            data = data[columns]
            data = data.replace([np.inf, -np.inf], np.nan).dropna()

            self.logger.info(f"Shape après préparation : {data.shape}")
            return data

        except Exception as e:
            self.logger.error(f"Erreur préparation données : {str(e)}")
            raise

    def detect_peaks(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Détecte les pics dans les données MS1 préparées.
        
        Args:
            data (pd.DataFrame): Données MS1 préparées
            
        Returns:
            pd.DataFrame: Pics détectés
        """
        try:
            self.logger.info("Construction des facteurs...")
            factors = deimos.build_factors(data, dims='detect')

            self.logger.info("Application du seuil...")
            data = deimos.threshold(data, threshold=self.peak_config.threshold)

            self.logger.info("Construction de l'index...")
            index = deimos.build_index(data, factors)

            self.logger.info("Lissage des données...")
            data = deimos.filters.smooth(
                data,
                index=index,
                dims=['mz', 'drift_time', 'retention_time'],
                radius=[
                    self.peak_config.smooth_radius['mz'],
                    self.peak_config.smooth_radius['drift_time'],
                    self.peak_config.smooth_radius['retention_time']
                ],
                iterations=self.peak_config.smooth_iterations
            )

            self.logger.info("Détection des pics...")
            peaks = deimos.peakpick.persistent_homology(
                data,
                index=index,
                dims=['mz', 'drift_time', 'retention_time'],
                radius=[
                    self.peak_config.peak_radius['mz'],
                    self.peak_config.peak_radius['drift_time'],
                    self.peak_config.peak_radius['retention_time']
                ]
            )

            peaks = peaks.sort_values(by='persistence', ascending=False).reset_index(drop=True)
            self.logger.info(f"Nombre de pics détectés : {len(peaks)}")
            return peaks

        except Exception as e:
            self.logger.error(f"Erreur détection pics : {str(e)}")
            raise

    def cluster_peaks(self, peaks_df: pd.DataFrame) -> pd.DataFrame:
        """
        Regroupe les pics similaires en utilisant DBSCAN.
        
        Args:
            peaks_df (pd.DataFrame): DataFrame des pics détectés
            
        Returns:
            pd.DataFrame: Pics regroupés
        """
        try:
            df = peaks_df    
            X = df[['mz', 'drift_time', 'retention_time']].to_numpy()
            median_mz = np.median(X[:, 0])
            
            # Calcul des tolérances selon la configuration
            mz_tolerance = median_mz * self.cluster_config.mz_ppm * 1e-6
            
            X_scaled = np.column_stack([
                X[:, 0] / mz_tolerance,
                X[:, 1] / self.cluster_config.dt_tolerance,
                X[:, 2] / self.cluster_config.rt_tolerance
            ])
            
            # Clustering avec DBSCAN
            clusters = DBSCAN(
                eps=self.cluster_config.dbscan_eps,
                min_samples=self.cluster_config.dbscan_min_samples,
                algorithm=self.cluster_config.algorithm,
                n_jobs=-1
            ).fit_predict(X_scaled)
            
            mask = clusters != -1
            df_valid = df[mask].copy()
            df_valid['cluster'] = clusters[mask]
            
            # Agrégation des clusters
            result = (df_valid.groupby('cluster')
                    .agg({
                        'intensity': ['idxmax', 'sum']
                    })
                    .reset_index())
            
            # Sélection des représentants
            representatives = df_valid.loc[result[('intensity', 'idxmax')]]
            representatives['intensity'] = result[('intensity', 'sum')].values
            
            # Tri final
            result_df = representatives.sort_values(
                by=["mz", "retention_time"]
            ).reset_index(drop=True)
            
            self.logger.info(f"Pics originaux : {len(peaks_df)}")
            self.logger.info(f"Pics après clustering : {len(result_df)}")
            
            return result_df

        except Exception as e:
            self.logger.error(f"Erreur clustering pics : {str(e)}")
            raise

    def process_sample(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Traite un échantillon complet : préparation, détection et clustering.
        
        Args:
            data (pd.DataFrame): Données brutes de l'échantillon
            
        Returns:
            Optional[pd.DataFrame]: Pics traités ou None si échec
        """
        try:
            # Préparation des données
            prepared_data = self.prepare_data(data)
            if prepared_data is None:
                return None
            
            # Détection des pics
            peaks = self.detect_peaks(prepared_data)
            if peaks.empty:
                self.logger.warning("Aucun pic détecté.")
                return None
            
            # Clustering des pics
            clustered_peaks = self.cluster_peaks(peaks)
            if clustered_peaks.empty:
                self.logger.warning("Aucun pic après clustering.")
                return None
                
            return clustered_peaks
            
        except Exception as e:
            self.logger.error(f"Erreur lors du traitement de l'échantillon : {str(e)}")
            return None