# scripts/quantification/compound_recovery.py

import pandas as pd
from pathlib import Path
from typing import List, Optional

def load_target_compounds(compounds_file: Path) -> List[str]:
    """Charge la liste des composés cibles pour les calibrants."""
    df = pd.read_csv(compounds_file)
    return df['Compound'].tolist()

def load_calibration_samples(calibration_file: Path) -> pd.DataFrame:
    """Charge les données des échantillons de calibration."""
    df = pd.read_csv(calibration_file)
    concentrations = pd.DataFrame({
        'Sample': df['Name'],
        'conc_M': df['conc_M']
    })
    return concentrations

def process_all_data(
    features_df: pd.DataFrame,
    feature_matrix_df: pd.DataFrame,
    target_compounds: List[str],
    calibration_df: pd.DataFrame
) -> pd.DataFrame:
    """Traite les données pour tous les échantillons:
    - Pour les échantillons de calibration: inclut composés cibles avec concentrations
    - Pour tous les échantillons: inclut toutes les molécules de niveau 1
    """
    results = []
    
    # On traite tous les composés de niveau 1
    level1_matches = features_df[features_df['confidence_level'] == 1].copy()
    
    for _, match_row in level1_matches.iterrows():
        compound = match_row['match_name']
        feature_id = f"{match_row['feature_id']}_mz{match_row['mz']:.4f}"
        samples = match_row['samples'].split(',')
        
        for sample in samples:
            sample = sample.strip()
            
            # On vérifie si c'est un échantillon de calibration
            is_calibration = sample in calibration_df['Sample'].values
            
            # Pour les composés cibles dans les échantillons de calibration, on récupère la concentration
            conc_val = None
            if is_calibration and compound in target_compounds:
                conc_val = calibration_df.loc[calibration_df['Sample'] == sample, 'conc_M'].iloc[0]
            
            intensity = feature_matrix_df.loc[sample, feature_id] if sample in feature_matrix_df.index and feature_id in feature_matrix_df.columns else None
            
            results.append({
                'Compound': compound,
                'SMILES': match_row['match_smiles'],
                'Feature_ID': feature_id,
                'Adduct': match_row['match_adduct'],
                'RT': match_row['retention_time'],
                'DT': match_row['drift_time'],
                'CCS': match_row['CCS'],
                'Total_Samples': len(samples),
                'Sample': sample,
                'Is_Calibration': is_calibration,
                'conc_M': conc_val,
                'Intensity': intensity,
                'Confidence_Level': match_row['confidence_level'],
                'daphnia_LC50_48_hr_ug/L': match_row.get('daphnia_LC50_48_hr_ug/L'),
                'algae_EC50_72_hr_ug/L': match_row.get('algae_EC50_72_hr_ug/L'),
                'pimephales_LC50_96_hr_ug/L': match_row.get('pimephales_LC50_96_hr_ug/L'),
                'Is_Target': compound in target_compounds
            })
    
    return pd.DataFrame(results)

def get_compound_summary(
    input_dir: Path,
    compounds_file: Path,
    calibration_file: Optional[Path] = None
) -> pd.DataFrame:
    """Génère un résumé complet des composés."""
    # Charger les données
    features_df = pd.read_parquet(input_dir / "feature_matrix/features_complete.parquet")
    feature_matrix_df = pd.read_parquet(input_dir / "feature_matrix/feature_matrix.parquet")
    
    # Charger la liste des composés cibles et les échantillons de calibration
    target_compounds = load_target_compounds(compounds_file)
    calibration_df = load_calibration_samples(calibration_file) if calibration_file else None
    
    if calibration_df is None:
        return pd.DataFrame()
    
    # Traiter toutes les données
    results_df = process_all_data(
        features_df,
        feature_matrix_df,
        target_compounds,
        calibration_df
    )
    
    if results_df.empty:
        return pd.DataFrame()
    
    columns = [
        'Compound', 'SMILES', 'Feature_ID', 'Adduct', 'RT', 'DT', 
        'CCS', 'Total_Samples', 'Sample', 'Is_Calibration', 'conc_M', 
        'Intensity', 'Confidence_Level', 'Is_Target',
        'daphnia_LC50_48_hr_ug/L', 'algae_EC50_72_hr_ug/L', 
        'pimephales_LC50_96_hr_ug/L'
    ]
    
    return results_df[columns].sort_values(['Compound', 'Sample', 'Adduct'])