# scripts/processing/peak_detection.py
import pandas as pd
import numpy as np
import deimos
from sklearn.cluster import DBSCAN
import logging

logger = logging.getLogger(__name__)

def prepare_data(df):
    """
    Prépare les données MS1.
    """
    try:
        # Filtrage MS1
        df['mslevel'] = df['mslevel'].astype(int)
        data = df[df['mslevel'] == 1].copy()
        
        if len(data) == 0:
            logger.warning("Aucune donnée MS1 trouvée")
            return None
        
        # Conversion des types
        for col in ['mz', 'intensity', 'rt', 'dt']:
            data[col] = data[col].astype(float)
        
        # Renommage des colonnes
        data = data.rename(columns={
            'rt': 'retention_time',
            'dt': 'drift_time',
            'intensity': 'intensity',
            'scanid': 'scanId'
        })
        
        # Sélection des colonnes
        columns = ['mz', 'intensity', 'drift_time', 'retention_time']
        data = data[columns]
        
        # Nettoyage
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna()
        
        logger.info(f"Shape après préparation: {data.shape}")
        return data
        
    except Exception as e:
        logger.error(f"Erreur préparation données: {str(e)}")
        raise

def detect_peaks(data):
    """
    Détecte les pics dans les données MS.
    """
    try:
        # Construction des facteurs
        logger.info("Construction des facteurs...")
        factors = deimos.build_factors(data, dims='detect')
        
        # Seuillage
        logger.info("Application du seuil...")
        data = deimos.threshold(data, threshold=100)
        
        # Construction de l'index
        logger.info("Construction de l'index...")
        index = deimos.build_index(data, factors)
        
        # Lissage
        logger.info("Lissage des données...")
        data = deimos.filters.smooth(
            data,
            index=index,
            dims=['mz', 'drift_time', 'retention_time'],
            radius=[0, 1, 0],
            iterations=7
        )
        
        # Détection des pics
        logger.info("Détection des pics...")
        peaks = deimos.peakpick.persistent_homology(
            data,
            index=index,
            dims=['mz', 'drift_time', 'retention_time'],
            radius=[2, 10, 0]
        )
        
        # Tri des pics
        peaks = peaks.sort_values(by='persistence', ascending=False).reset_index(drop=True)
        logger.info(f"Nombre de pics détectés : {len(peaks)}")
        
        return peaks
        
    except Exception as e:
        logger.error(f"Erreur détection pics: {str(e)}")
        raise

def cluster_peaks(peaks_df):
    """
    Clustering des pics similaires.
    """
    try:
        # Copie et préparation
        df = peaks_df.copy()
        X = df[['mz', 'drift_time', 'retention_time']].values
        
        # Tolérances
        mz_tolerance = 1e-4
        dt_tolerance = 0.10
        rt_tolerance = 0.20
        
        # Calcul des eps
        eps_mz = np.median(X[:, 0]) * mz_tolerance
        eps_dt = np.median(X[:, 1]) * dt_tolerance
        eps_rt = np.median(X[:, 2]) * rt_tolerance
        
        # Normalisation
        X_scaled = np.zeros_like(X)
        X_scaled[:, 0] = X[:, 0] / eps_mz
        X_scaled[:, 1] = X[:, 1] / eps_dt
        X_scaled[:, 2] = X[:, 2] / eps_rt
        
        # DBSCAN
        clusters = DBSCAN(eps=1.0, min_samples=2).fit_predict(X_scaled)
        df['cluster'] = clusters
        
        # Agrégation
        result = []
        for cluster_id in sorted(set(clusters)):
            cluster_data = df[df['cluster'] == cluster_id]
            if cluster_id == -1:
                result.append(cluster_data)
            else:
                max_intensity_idx = cluster_data['intensity'].idxmax()
                max_intensity_row = cluster_data.loc[max_intensity_idx].copy()
                max_intensity_row['cluster_size'] = len(cluster_data)
                result.append(pd.DataFrame([max_intensity_row]))
        
        result = pd.concat(result, ignore_index=True)
        result = result.sort_values('intensity', ascending=False).reset_index(drop=True)
        
        logger.info(f"Pics original : {len(peaks_df)}")
        logger.info(f"Pics après clustering : {len(result)}")
        
        return result
        
    except Exception as e:
        logger.error(f"Erreur clustering pics: {str(e)}")
        raise
