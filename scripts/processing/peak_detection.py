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
    
    df = peaks_df    
    X = df[['mz', 'drift_time', 'retention_time']].to_numpy()
    median_mz = np.median(X[:, 0])
    # Calcul des tolérances
    mz_tolerance = median_mz * 10e-6
    X_scaled = np.column_stack([
        X[:, 0] / mz_tolerance,
        X[:, 1] / 1.0,  # dt_tolerance
        X[:, 2] / 0.1   # rt_tolerance
    ])
    
    # DBSCAN  avec algorithm='ball_tree' pour données 3D
    clusters = DBSCAN(
        eps=1.0, 
        min_samples=1,
        algorithm='ball_tree',
        n_jobs=-1  # Parrallélisation
    ).fit_predict(X_scaled)
    
    mask = clusters != -1
    df_valid = df[mask].copy()
    df_valid['cluster'] = clusters[mask]
    
    result = (df_valid.groupby('cluster')
              .agg({
                  'intensity': ['idxmax', 'sum']
              })
              .reset_index())
    
    # Récupération des représentants
    representatives = df_valid.loc[result[('intensity', 'idxmax')]]
    representatives['intensity'] = result[('intensity', 'sum')].values
    
    # Tri final
    result_df = representatives.sort_values(
        by=["mz", "retention_time"]
    ).reset_index(drop=True)
    
    logger.info(f"Pics originaux : {len(peaks_df)}")
    logger.info(f"Pics après clustering : {len(result_df)}")
    
    return result_df



def cluster_peaks_bijection(peaks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Regroupe les pics similaires en clusters en utilisant des tolérances définies.

    Args:
        peaks_df (pd.DataFrame): DataFrame contenant les pics détectés.

    Returns:
        pd.DataFrame: DataFrame avec un pic représentatif par cluster.

    Raises:
        Exception: Si une erreur survient pendant le clustering.
    """
    def quotient_compute(a: float, b: float) -> float:
        """
        Calcule un quotient relatif représentant l'écart proportionnel entre deux valeurs.

        Args:
            a (float): La première valeur.
            b (float): La deuxième valeur.

        Returns:
            float: Le quotient proportionnel représentant l'écart entre `a` et `b`.

        Raises:
            ValueError: Si une division par zéro est détectée.
        """
        # Vérifie pour éviter la division par zéro
        if b == 0 or a == 0:
            raise ValueError("Une division par zéro n'est pas possible.")

        # Calcule l'écart proportionnel
        return 1 - (a / b) if a < b else 1 - (b / a)

    def compute_distance(row: np.ndarray, candidate: np.ndarray, dims: list, bij_tables: list) -> float:
        """
        Calcule la distance euclidienne entre deux points dans des dimensions spécifiées.

        Args:
            row (np.ndarray): Point de référence sous forme de tableau NumPy.
            candidate (np.ndarray): Point candidat sous forme de tableau NumPy.
            dims (list): Liste des dimensions utilisées pour le calcul de la distance.
            bij_tables (list): Liste de tables de bijection pour mapper les indices des colonnes.

        Returns:
            float: La distance euclidienne entre `row` et `candidate`.
        """
        # Cas où une seule table de bijection est fournie
        if len(bij_tables) == 1:
            return np.sqrt(sum(
                (row[bij_tables[0][dim]] - candidate[bij_tables[0][dim]])**2
                for dim in dims
            ))

        # Cas où deux tables de bijection sont fournies
        return np.sqrt(sum(
            (row[bij_tables[0][dim]] - candidate[bij_tables[1][dim]])**2
            for dim in dims
        ))

    try:
        # Crée une copie du DataFrame pour éviter de modifier les données originales
        df = peaks_df.copy()

        # Définit les tolérances pour le clustering
        tolerances = {"mz": 1e-4, "retention_time": 0.10, "drift_time": 0.20}

        # Initialise les colonnes nécessaires pour le clustering
        df["cluster"] = -1  # Clusters initialisés à -1 (non assigné)
        df["distance"] = np.inf  # Distances initialisées à l'infini
        df = df.sort_values(by=["intensity"], ascending=False).reset_index(drop=True)

        # Convertit le DataFrame en tableau NumPy pour des accès rapides
        df_array = df.to_numpy()

        # Crée des tables de bijection pour mapper les dimensions
        tl_bijection = {dim: idx for idx, dim in enumerate(tolerances.keys())}
        df_bijection = {dim: idx for idx, dim in enumerate(df.columns)}

        # Initialise le premier identifiant de cluster
        cluster_id = 0

        # Parcourt chaque point dans le tableau NumPy
        for i, row in enumerate(df_array):
            if row[-2] == -1:  # Vérifie si le point n'est pas encore assigné
                row[-2] = cluster_id  # Assigne le cluster courant

                for j, candidate in enumerate(df_array):
                    if i != j:  # Ignore le même point
                        # Vérifie si le candidat respecte les tolérances
                        is_within_threshold = all(
                            quotient_compute(
                                a=row[df_bijection[dim]],
                                b=candidate[df_bijection[dim]]
                            ) <= tolerances[dim]
                            for dim in tolerances.keys()
                        )

                        if is_within_threshold:
                            # Calcule la distance entre le point et le candidat
                            distance = compute_distance(
                                row=row,
                                candidate=candidate,
                                dims=list(tl_bijection.keys()),
                                bij_tables=[df_bijection]
                            )

                            # Met à jour le cluster et la distance si nécessaire
                            if distance < df_array[j, -1]:
                                df_array[j, -2] = cluster_id
                                df_array[j, -1] = distance

                # Passe au cluster suivant
                cluster_id += 1

        # Reconstruit un DataFrame à partir du tableau NumPy modifié
        df_array = pd.DataFrame(data=df_array, columns=df.columns)

        # Conserve uniquement les points avec l'intensité maximale dans chaque cluster
        df_array = df_array.loc[df_array.groupby("cluster")["intensity"].idxmax()]

        # Trie les résultats et supprime les colonnes inutiles
        df_array = df_array.drop(columns=["cluster", "distance"]).sort_values(
            by=["mz", "retention_time"], ascending=True
        ).reset_index(drop=True)

        # Log des statistiques sur le clustering
        logger.info(f"Pics originaux : {len(peaks_df)}")
        logger.info(f"Pics après clustering : {len(df_array)}")

        return df_array

    except Exception as e:
        # Log toute erreur survenue
        logger.error(f"Erreur clustering pics : {str(e)}")
        raise
