# scripts/utils/matching_utils.py

import pandas as pd
import numpy as np
from ..config.config import Config

def calculate_match_scores(match, tolerances=None):
    if tolerances is None:
        tolerances = {
            'mz_ppm': 5,
            'ccs_percent': 8,
            'rt_min': 2
        }
    
    weights = {'mz': 0.4, 'ccs': 0.4, 'rt': 0.2}
    scores = {}

    # Score m/z - toujours calculé
    scores['mz'] = max(0, 1 - abs(match['mz_error_ppm']) / tolerances['mz_ppm'])

    # Score CCS - hiérarchie exp > pred
    if pd.notna(match['match_ccs_exp']):
        ccs_error = abs(match['ccs_error_percent'])
        ccs_source = 'exp'
        weights['ccs'] *= 1.2
        scores['ccs'] = max(0, 1 - ccs_error / tolerances['ccs_percent'])
    elif pd.notna(match['match_ccs_pred']):
        match['ccs_error_percent'] = (match['peak_ccs'] - match['match_ccs_pred']) / match['match_ccs_pred'] * 100
        ccs_error = abs(match['ccs_error_percent'])
        ccs_source = 'pred'
        weights['ccs'] *= 0.6
        scores['ccs'] = max(0, 1 - ccs_error / tolerances['ccs_percent'])
    else:
        scores['ccs'] = 0
        ccs_source = None

    # Score RT - hiérarchie obs > pred
    if pd.notna(match['match_rt_obs']):
        rt_error = abs(match['rt_error_min'])
        rt_source = 'obs'
        weights['rt'] *= 1.2
        scores['rt'] = max(0, 1 - rt_error / tolerances['rt_min'])
    elif pd.notna(match['match_rt_pred']):
        match['rt_error_min'] = abs(match['peak_rt'] - match['match_rt_pred'])
        rt_error = match['rt_error_min']
        rt_source = 'pred'
        weights['rt'] *= 0.6
        scores['rt'] = max(0, 1 - rt_error / tolerances['rt_min'])
    else:
        scores['rt'] = 0
        rt_source = None

    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}
    global_score = sum(scores[key] * weights[key] for key in weights)

    return {
        'individual_scores': scores,
        'global_score': global_score,
        'ccs_source': ccs_source,
        'rt_source': rt_source
    }

def find_matches_asof(peaks_df, db_df, tolerances=None):
    """Méthode utilisant merge_asof"""
    if tolerances is None:
        tolerances = Config.IDENTIFICATION['tolerances']
        
    # Trier les DataFrames
    peaks_df = peaks_df.sort_values('mz')
    db_df = db_df.sort_values('mz')
    
    # Calcul de la tolérance m/z en valeur absolue
    mz_tolerance = peaks_df['mz'].mean() * tolerances['mz_ppm'] * 1e-6
    
    # Merge asof
    matches = pd.merge_asof(
        peaks_df,
        db_df,
        on='mz',
        direction='nearest',
        tolerance=mz_tolerance
    )
    
    return matches
    
    
def assign_confidence_level(match, tolerances=None):
    """Assigne un niveau de confiance basé sur les critères disponibles"""
    if tolerances is None:
        tolerances = {
            'mz_ppm': 5,           # Tolérance m/z réduite à 5 ppm
            'ccs_exp': 8,          # CCS expérimentale
            'ccs_exp_l2': 8,       
            'ccs_exp_l3': 8,       
            'ccs_pred': 8,         
            'rt_obs_l1': 0.5,      
            'rt_obs_l2': 1.0,      
            'rt_obs_l3': 2.0,      
            'rt_pred': 3.0
        }

    # Vérification m/z - maintenant 5 ppm
    mz_ok = abs(match['mz_error_ppm']) <= tolerances['mz_ppm']
    if not mz_ok:
        return 5, "Match m/z hors tolérance (5 ppm)"

    # Vérification disponibilité des données
    has_ccs_exp = pd.notna(match['match_ccs_exp'])
    has_ccs_pred = pd.notna(match['match_ccs_pred'])
    has_rt_obs = pd.notna(match['match_rt_obs'])
    has_rt_pred = pd.notna(match['match_rt_pred'])

    # Niveau 1: CCS exp + RT obs dans tolérances strictes
    if has_ccs_exp and has_rt_obs:
        if (abs(match['ccs_error_percent']) <= tolerances['ccs_exp'] and 
            abs(match['rt_error_min']) <= tolerances['rt_obs_l1']):
            return 1, "Match parfait (CCS exp + RT obs)"

    # Niveau 2: CCS exp + RT obs dans tolérances élargies
    if has_ccs_exp and has_rt_obs:
        if (abs(match['ccs_error_percent']) <= tolerances['ccs_exp_l2'] and 
            abs(match['rt_error_min']) <= tolerances['rt_obs_l2']):
            return 2, "Match très probable (CCS exp + RT obs)"

    # Niveau 3: CCS exp ou pred + RT pred ou obs
    if has_ccs_exp and has_rt_obs:
        return 3, "Match probable (CCS exp + RT obs hors tolérance)"
    elif has_ccs_exp and has_rt_pred:
        return 3, "Match probable (CCS exp + RT pred)"
    elif has_ccs_pred and has_rt_obs:
        if abs(match['rt_error_min']) <= tolerances['rt_obs_l3']:
            return 3, "Match probable (CCS pred + RT obs)"

    # Niveau 4: CCS pred disponible
    if has_ccs_pred:
        if has_rt_pred:
            return 4, "Match possible (CCS pred + RT pred)"
        return 4, "Match possible (CCS pred uniquement)"

    # Niveau 5: Uniquement match m/z
    return 5, "Match incertain (m/z uniquement)"



def find_matches_window(peaks_df, db_df, tolerances=None):
    if tolerances is None:
        tolerances = {
            'mz_ppm': 5,           # Tolérance en ppm pour la valeur m/z
            'ccs_percent': 8,      # Tolérance en pourcentage pour la CCS
            'rt_min': 2            # Tolérance en minutes pour le temps de rétention
        }
        
    # Création de clés composites dans 'db_df' pour 'Name', 'adduct' et 'SMILES'
    db_df['Name_str'] = db_df['Name'].astype(str).fillna('')
    db_df['adduct_str'] = db_df['adduct'].astype(str).fillna('')
    db_df['SMILES_str'] = db_df['SMILES'].astype(str).fillna('')
    
    # Utilisation de 'Name' et 'adduct' comme identifiant moléculaire si disponibles, sinon utiliser 'SMILES'
    db_df['molecule_id'] = db_df.apply(
        lambda row: f"{row['Name_str']}_{row['adduct_str']}" if row['Name_str'] and row['adduct_str'] else row['SMILES_str'],
        axis=1
    )
    
    # Calcul de 'has_ms2_db' basé sur la disponibilité des données MS2 pour 'molecule_id'
    db_df['has_ms2_db'] = db_df['peaks_ms2_mz'].apply(lambda x: 1 if isinstance(x, list) else 0)
    ms2_df = db_df.groupby('molecule_id')['has_ms2_db'].max().reset_index()
    db_df = db_df.drop(columns=['has_ms2_db'])
    db_df = db_df.merge(ms2_df, on='molecule_id', how='left')
    
    # Tri des données de la base par 'mz' et récupération des valeurs 'mz' sous forme de tableau NumPy
    db_df = db_df.sort_values('mz').reset_index(drop=True)
    db_mz = db_df['mz'].values

    all_matches = []

    for peak in peaks_df.itertuples():
        # Calcul des limites de tolérance pour le m/z
        mz_tolerance = peak.mz * tolerances['mz_ppm'] * 1e-6
        mz_min = peak.mz - mz_tolerance
        mz_max = peak.mz + mz_tolerance

        # Recherche efficace des indices correspondants
        idx_start = np.searchsorted(db_mz, mz_min, side='left')
        idx_end = np.searchsorted(db_mz, mz_max, side='right')

        matches = db_df.iloc[idx_start:idx_end]

        if not matches.empty:
            for match in matches.itertuples():
                # Calcul et validation des erreurs pour le temps de rétention
                rt_match = False
                rt_error = None
                if pd.notna(match.Observed_RT):
                    rt_error = abs(peak.retention_time - match.Observed_RT)
                    rt_match = rt_error <= tolerances['rt_min']
                elif pd.notna(match.Predicted_RT):
                    rt_error = abs(peak.retention_time - match.Predicted_RT)
                    rt_match = rt_error <= tolerances['rt_min']

                # Calcul et validation des erreurs pour la CCS
                ccs_match = False
                ccs_error = None
                if pd.notna(match.ccs_exp):
                    ccs_error = abs((peak.CCS - match.ccs_exp) / match.ccs_exp * 100)
                    ccs_match = ccs_error <= tolerances['ccs_percent']
                elif pd.notna(match.ccs_pred):
                    ccs_error = abs((peak.CCS - match.ccs_pred) / match.ccs_pred * 100)
                    ccs_match = ccs_error <= tolerances['ccs_percent']

                # Ajout des correspondances si elles respectent les tolérances
                if (rt_match or rt_error is None) and (ccs_match or ccs_error is None):
                    match_details = {
                        'peak_mz': peak.mz,
                        'peak_rt': peak.retention_time,
                        'peak_dt': peak.drift_time,
                        'peak_intensity': peak.intensity,
                        'peak_ccs': peak.CCS,
                        'match_name': match.Name,
                        'match_adduct': match.adduct,
                        'match_smiles': match.SMILES,
                        'match_mz': match.mz,
                        'mz_error_ppm': (peak.mz - match.mz) / match.mz * 1e6,
                        'match_ccs_exp': match.ccs_exp,
                        'match_ccs_pred': match.ccs_pred,
                        'match_rt_obs': match.Observed_RT,
                        'match_rt_pred': match.Predicted_RT,
                        'rt_error_min': rt_error,
                        'ccs_error_percent': ccs_error,
                        'has_ms2_db': match.has_ms2_db,
                        'molecule_id': match.molecule_id
                    }

                    # Calcul des scores de correspondance et niveaux de confiance
                    score_details = calculate_match_scores(match_details)  # Fonction définie ailleurs
                    confidence_level, confidence_reason = assign_confidence_level(match_details)  # Fonction définie ailleurs
                    match_details.update({
                        'individual_scores': score_details['individual_scores'],
                        'global_score': score_details['global_score'],
                        'ccs_source': score_details['ccs_source'],
                        'rt_source': score_details['rt_source'],
                        'confidence_level': confidence_level,
                        'confidence_reason': confidence_reason
                    })

                    all_matches.append(match_details)

    # Conversion des résultats en DataFrame
    matches_df = pd.DataFrame(all_matches) if all_matches else pd.DataFrame()

    if not matches_df.empty:
        # Tri des correspondances pour prioriser 'has_ms2_db' et le score global
        matches_df = matches_df.sort_values(
            ['molecule_id', 'has_ms2_db', 'global_score'],
            ascending=[True, False, False]
        )
        # Suppression des doublons en gardant les correspondances avec 'has_ms2_db' == 1 si disponibles
        matches_df = matches_df.drop_duplicates(subset='molecule_id', keep='first')

        # Tri final selon le niveau de confiance et le score global
        matches_df = matches_df.sort_values(['confidence_level', 'global_score'], ascending=[True, False])

    return matches_df

