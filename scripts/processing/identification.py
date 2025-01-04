#scripts/processing/identification.py
#-*- coding:utf-8 -*-

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm
from scipy.spatial.distance import cdist
from ..config.config import Config  
from ..utils.matching_utils import calculate_match_scores, assign_confidence_level 

class CompoundIdentifier:
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.config = Config.IDENTIFICATION
        self.db = pd.DataFrame()
        # Utiliser 75% des cœurs disponibles
        self.n_workers = max(1, int(mp.cpu_count() * 0.75))
        self.load_database()

    def load_database(self) -> None:
        try:
            db_path = Path(Config.PATHS.INPUT_DATABASES) / self.config.database_file
            if not db_path.exists():
                raise FileNotFoundError(f"Base de données non trouvée: {db_path}")

            # Chargement optimisé avec types spécifiés
            dtype_map = {
                'mz': np.float32,
                'ccs_exp': np.float32,
                'ccs_pred': np.float32,
                'Observed_RT': np.float32,
                'Predicted_RT': np.float32
            }
            
            self.db = pd.read_hdf(db_path, key=self.config.database_key)
            
            # Conversion des types pour optimisation
            for col, dtype in dtype_map.items():
                if col in self.db.columns:
                    self.db[col] = self.db[col].astype(dtype)

            # Pré-traitement des données MS2
            if 'peaks_ms2_mz' in self.db.columns and 'peaks_ms2_intensities' in self.db.columns:
                for col in ['peaks_ms2_mz', 'peaks_ms2_intensities']:
                    self.db[col] = self.db[col].apply(self._convert_peaks_string_to_list)
                    
            # Pré-calcul des identifiants uniques
            self.db['Name_str'] = self.db['Name'].astype(str).fillna('')
            self.db['adduct_str'] = self.db['adduct'].astype(str).fillna('')
            self.db['molecule_id'] = self.db['Name_str'] + '_' + self.db['adduct_str']
            
            # Pré-calcul du statut MS2
            self.db['has_ms2_db'] = self.db['peaks_ms2_mz'].apply(
                lambda x: 1 if isinstance(x, list) and len(x) > 0 else 0
            )

            self.logger.info(f"Base de données chargée avec succès : {len(self.db)} composés")

        except Exception as e:
            self.logger.error(f"Erreur lors du chargement de la base de données : {str(e)}")
            raise

    def _convert_peaks_string_to_list(self, peaks_str: str) -> list:
        try:
            if pd.isna(peaks_str):
                return []
            if isinstance(peaks_str, list):
                return peaks_str
            if isinstance(peaks_str, str):
                peaks_str = peaks_str.strip('[]')
                if not peaks_str:
                    return []
                return [float(x) for x in peaks_str.split(',')]
            return []
        except Exception as e:
            self.logger.warning(f"Erreur de conversion des pics: {str(e)}")
            return []

    def find_matches_vectorized(self, peaks_df: pd.DataFrame, tolerances: Dict[str, float]) -> pd.DataFrame:
        """Version vectorisée de la recherche des correspondances."""
        try:
            # Conversion en arrays NumPy pour optimisation
            peak_mz = peaks_df['mz'].values.astype(np.float32)
            peak_rt = peaks_df['retention_time'].values.astype(np.float32)
            peak_ccs = peaks_df['CCS'].values.astype(np.float32)
            
            db_mz = self.db['mz'].values.astype(np.float32)
            
            # Calcul vectorisé des différences de m/z
            mz_tolerance = np.outer(peak_mz, np.ones_like(db_mz)) * tolerances['mz_ppm'] * 1e-6
            mz_diff_matrix = np.abs(np.subtract.outer(peak_mz, db_mz))
            valid_mz_mask = mz_diff_matrix <= mz_tolerance
            
            all_matches = []
            
            # Traitement par lots pour éviter la surcharge mémoire
            batch_size = 1000
            for i in range(0, len(peaks_df), batch_size):
                batch_mask = valid_mz_mask[i:i+batch_size]
                batch_peaks = peaks_df.iloc[i:i+batch_size]
                
                # Pour chaque pic dans le lot
                for peak_idx, peak_matches in enumerate(batch_mask):
                    peak = batch_peaks.iloc[peak_idx]
                    matched_db_indices = np.where(peak_matches)[0]
                    
                    if len(matched_db_indices) == 0:
                        continue
                        
                    # Traitement vectorisé des correspondances
                    matched_db = self.db.iloc[matched_db_indices]
                    
                    # Calcul vectorisé des erreurs
                    rt_errors = np.full(len(matched_db), np.inf)
                    ccs_errors = np.full(len(matched_db), np.inf)
                    
                    # RT errors
                    obs_rt_mask = pd.notna(matched_db['Observed_RT'])
                    pred_rt_mask = pd.notna(matched_db['Predicted_RT'])
                    
                    if obs_rt_mask.any():
                        rt_errors[obs_rt_mask] = np.abs(
                            peak.retention_time - matched_db.loc[obs_rt_mask, 'Observed_RT']
                        )
                    if pred_rt_mask.any():
                        rt_errors[pred_rt_mask] = np.minimum(
                            rt_errors[pred_rt_mask],
                            np.abs(peak.retention_time - matched_db.loc[pred_rt_mask, 'Predicted_RT'])
                        )
                    
                    # CCS errors
                    exp_ccs_mask = pd.notna(matched_db['ccs_exp'])
                    pred_ccs_mask = pd.notna(matched_db['ccs_pred'])
                    
                    if exp_ccs_mask.any():
                        ccs_errors[exp_ccs_mask] = np.abs(
                            (peak.CCS - matched_db.loc[exp_ccs_mask, 'ccs_exp']) / 
                            matched_db.loc[exp_ccs_mask, 'ccs_exp'] * 100
                        )
                    if pred_ccs_mask.any():
                        ccs_errors[pred_ccs_mask] = np.minimum(
                            ccs_errors[pred_ccs_mask],
                            np.abs((peak.CCS - matched_db.loc[pred_ccs_mask, 'ccs_pred']) / 
                                 matched_db.loc[pred_ccs_mask, 'ccs_pred'] * 100)
                        )
                    
                    # Filtrage final
                    valid_matches = (
                        (rt_errors <= tolerances['rt_min']) | 
                        (rt_errors == np.inf)
                    ) & (
                        (ccs_errors <= tolerances['ccs_percent']) | 
                        (ccs_errors == np.inf)
                    )
                    
                    if not valid_matches.any():
                        continue
                        
                    # Création des correspondances
                    for match_idx in np.where(valid_matches)[0]:
                        match = matched_db.iloc[match_idx]
                        
                        match_details = {
                            'peak_mz': peak.mz,
                            'peak_rt': peak.retention_time,
                            'peak_dt': peak.drift_time,
                            'peak_intensity': peak.intensity,
                            'peak_ccs': peak.CCS,
                            'match_name': match.Name,
                            'match_adduct': match.adduct,
                            'match_smiles': match.SMILES,
                            'categories': match.categories,
                            'match_mz': match.mz,
                            'mz_error_ppm': (peak.mz - match.mz) / match.mz * 1e6,
                            'match_ccs_exp': match.ccs_exp,
                            'match_ccs_pred': match.ccs_pred,
                            'match_rt_obs': match.Observed_RT,
                            'match_rt_pred': match.Predicted_RT,
                            'rt_error_min': rt_errors[match_idx] if rt_errors[match_idx] != np.inf else None,
                            'ccs_error_percent': ccs_errors[match_idx] if ccs_errors[match_idx] != np.inf else None,
                            'has_ms2_db': match.has_ms2_db,
                            'molecule_id': match.molecule_id
                        }
                        
                        # Ajout des scores toxicologiques si présents
                        tox_columns = ['LC50_48_hr_ug/L', 'EC50_72_hr_ug/L', 'LC50_96_hr_ug/L']
                        for col in tox_columns:
                            if col in match and pd.notna(match[col]):
                                match_details[f"daphnia_{col}" if '48' in col else
                                            f"algae_{col}" if '72' in col else
                                            f"pimephales_{col}"] = float(match[col])
                            else:
                                match_details[f"daphnia_{col}" if '48' in col else
                                            f"algae_{col}" if '72' in col else
                                            f"pimephales_{col}"] = None
                        
                        # Calcul des scores et niveau de confiance
                        score_details = calculate_match_scores(match_details, tolerances)
                        confidence_level, confidence_reason = assign_confidence_level(match_details, tolerances)
                        
                        match_details.update({
                            'individual_scores': score_details['individual_scores'],
                            'global_score': score_details['global_score'],
                            'ccs_source': score_details['ccs_source'],
                            'rt_source': score_details['rt_source'],
                            'confidence_level': confidence_level,
                            'confidence_reason': confidence_reason
                        })
                        
                        all_matches.append(match_details)
            
            # Création du DataFrame final
            matches_df = pd.DataFrame(all_matches) if all_matches else pd.DataFrame()
            
            if not matches_df.empty:
                # Trie et dédoublonnage optimisés
                matches_df['sort_key'] = matches_df['has_ms2_db'].astype(str) + '_' + \
                                       matches_df['global_score'].astype(str)
                                       
                matches_df = matches_df.sort_values(
                    'sort_key', ascending=False
                ).drop_duplicates(
                    subset='molecule_id', keep='first'
                ).drop(columns=['sort_key'])
                
                matches_df = matches_df.sort_values(
                    ['confidence_level', 'global_score'], 
                    ascending=[True, False]
                )
            
            return matches_df
            
        except Exception as e:
            self.logger.error(f"Erreur dans la recherche vectorisée : {str(e)}")
            raise

    def identify_compounds(self, peaks_df: pd.DataFrame, output_dir: str) -> Optional[pd.DataFrame]:
        self.logger.info("Début du processus d'identification des composés.")
        
        try:
            # Conversion des types pour optimisation
            peaks_df['mz'] = peaks_df['mz'].astype(np.float32)
            peaks_df['retention_time'] = peaks_df['retention_time'].astype(np.float32)
            peaks_df['CCS'] = peaks_df['CCS'].astype(np.float32)
            
            # Recherche vectorisée des correspondances
            matches_df = self.find_matches_vectorized(
                peaks_df=peaks_df,
                tolerances=self.config.tolerances
            )

            if matches_df.empty:
                self.logger.warning("Aucune correspondance trouvée.")
                return None

            # Sauvegarde des résultats
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            matches_path = output_path / 'all_matches.parquet'
            matches_df.to_parquet(matches_path)

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