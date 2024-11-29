#scripts/processing/peak_detection.py
#-*- coding:utf-8 -*-

import deimos
import logging
import numpy as np
import pandas as pd
from typing import Optional
from sklearn.cluster import DBSCAN

# Initialiser le logger
logger = logging.getLogger(__name__)

def prepare_data(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Prépare les données MS1 pour la détection de pics.
    """
    try:
        df['mslevel'] = df['mslevel'].astype(int)
        data = df[df['mslevel'] == 1].copy()

        if len(data) == 0:
            logger.warning("Aucune donnée MS1 trouvée.")
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

        logger.info(f"Shape après préparation : {data.shape}")
        return data

    except Exception as e:
        logger.error(f"Erreur préparation données : {str(e)}")
        raise

def detect_peaks(data: pd.DataFrame) -> pd.DataFrame:
    """
    Détecte les pics dans les données MS1 préparées.
    """
    try:
        logger.info("Construction des facteurs...")
        factors = deimos.build_factors(data, dims='detect')

        logger.info("Application du seuil...")
        data = deimos.threshold(data, threshold=100)

        logger.info("Construction de l'index...")
        index = deimos.build_index(data, factors)

        logger.info("Lissage des données...")
        data = deimos.filters.smooth(
            data,
            index=index,
            dims=['mz', 'drift_time', 'retention_time'],
            radius=[0, 1, 0],
            iterations=7
        )

        logger.info("Détection des pics...")
        peaks = deimos.peakpick.persistent_homology(
            data,
            index=index,
            dims=['mz', 'drift_time', 'retention_time'],
            radius=[2, 10, 0]
        )

        peaks = peaks.sort_values(by='persistence', ascending=False).reset_index(drop=True)
        logger.info(f"Nombre de pics détectés : {len(peaks)}")
        return peaks

    except Exception as e:
        logger.error(f"Erreur détection pics : {str(e)}")
        raise

def cluster_peaks(peaks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Regroupe les pics similaires en clusters en utilisant DBSCAN.
    """
    try:
        df = peaks_df.copy()
        X = df[['mz', 'drift_time', 'retention_time']].values
        
        mz_tolerance = np.median(X[:, 0]) * 1e-4
        dt_tolerance = np.median(X[:, 1]) * 0.10
        rt_tolerance = 0.20
        
        X_scaled = np.zeros_like(X)
        X_scaled[:, 0] = X[:, 0] / mz_tolerance
        X_scaled[:, 1] = X[:, 1] / dt_tolerance
        X_scaled[:, 2] = X[:, 2] / rt_tolerance
        
        clusters = DBSCAN(eps=1.0, min_samples=1).fit_predict(X_scaled)
        df['cluster'] = clusters
        
        result = []
        for cluster_id in sorted(set(clusters)):
            if cluster_id == -1:
                continue
                
            cluster_data = df[df['cluster'] == cluster_id]
            max_intensity_idx = cluster_data['intensity'].idxmax()
            representative = cluster_data.loc[max_intensity_idx].copy()
            representative = representative.drop('cluster')
            result.append(representative)
        
        result_df = pd.DataFrame(result) if result else pd.DataFrame()
        
        if not result_df.empty:
            result_df = result_df.sort_values(
                by=["mz", "retention_time"], 
                ascending=True
            ).reset_index(drop=True)
        
        logger.info(f"Pics originaux : {len(peaks_df)}")
        logger.info(f"Pics après clustering : {len(result_df)}")
        
        return result_df

    except Exception as e:
        logger.error(f"Erreur clustering pics : {str(e)}")
        raise