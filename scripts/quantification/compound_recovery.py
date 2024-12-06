# scripts/quantification/compound_recovery.py

import pandas as pd
from pathlib import Path
from typing import List, Optional

def load_target_compounds(compounds_file: Path) -> List[str]:
    """Charge la liste des composés cibles."""
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

def get_best_adduct_intensities(
    features_df: pd.DataFrame,
    feature_matrix_df: pd.DataFrame,
    compounds: List[str],
    calibration_df: Optional[pd.DataFrame] = None,
    min_samples: int = 4
) -> pd.DataFrame:
    """Récupère les intensités de l'adduit le plus intense pour chaque composé."""
    results = []
    
    for compound in compounds:
        compound_matches = features_df[features_df['match_name'] == compound].copy()
        
        if compound_matches.empty:
            continue
            
        compound_matches['n_samples'] = compound_matches['samples'].str.count(',') + 1
        compound_matches = compound_matches[compound_matches['n_samples'] >= min_samples]
        
        if compound_matches.empty:
            continue
            
        best_match_idx = compound_matches['intensity'].idxmax()
        best_match = compound_matches.loc[best_match_idx]
        feature_id = f"{best_match['feature_id']}_mz{best_match['mz']:.4f}"
        
        for sample in best_match['samples'].split(','):
            sample = sample.strip()
            # Vérifier si c'est un échantillon de calibration
            if calibration_df is not None and sample in calibration_df['Sample'].values:
                intensity = feature_matrix_df.loc[sample, feature_id]
                
                results.append({
                    'Compound': compound,
                    'SMILES': best_match['match_smiles'],
                    'Feature_ID': feature_id,
                    'Adduct': best_match['match_adduct'],
                    'RT': best_match['retention_time'],
                    'DT': best_match['drift_time'],
                    'CCS': best_match['CCS'],
                    'Total_Samples': best_match['n_samples'],
                    'Sample': sample,
                    'conc_M': calibration_df.loc[calibration_df['Sample'] == sample, 'conc_M'].iloc[0],
                    'Intensity': intensity,
                    'Confidence_Level': best_match['confidence_level']
                })
    
    if not results:
        return pd.DataFrame()
        
    df = pd.DataFrame(results)
    
    columns = ['Compound', 'SMILES', 'Feature_ID', 'Adduct', 'RT', 'DT', 
               'CCS', 'Total_Samples', 'Sample', 'conc_M', 'Intensity', 'Confidence_Level']
    
    return df[columns]

def get_compound_summary(
    input_dir: Path,
    compounds_file: Path,
    calibration_file: Optional[Path] = None,
    min_samples: int = 4
) -> pd.DataFrame:
    """Génère un résumé des composés trouvés avec leurs meilleurs adduits."""
    features_df = pd.read_parquet(input_dir / "feature_matrix/features_complete.parquet")
    feature_matrix_df = pd.read_parquet(input_dir / "feature_matrix/feature_matrix.parquet")
    compounds = load_target_compounds(compounds_file)
    
    calibration_df = None
    if calibration_file is not None:
        calibration_df = load_calibration_samples(calibration_file)
    
    summary_df = get_best_adduct_intensities(
        features_df,
        feature_matrix_df,
        compounds,
        calibration_df,
        min_samples
    )
    
    if not summary_df.empty:
        summary_df = summary_df.sort_values(['Compound', 'conc_M'])
    
    return summary_df