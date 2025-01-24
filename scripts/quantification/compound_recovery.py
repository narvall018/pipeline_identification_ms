import pandas as pd
from pathlib import Path
from typing import List, Optional
import pyarrow as pa
import pyarrow.parquet as pq

def get_mass_difference(adduct: str) -> float:
    """Retourne la différence de masse pour chaque adduit."""
    adduct_masses = {
        "[M+H]+": 1.007825,
        "[M+Na]+": 22.989769,
        "[M+NH4]+": 18.034374
    }
    return adduct_masses.get(adduct, 1.007825)  # Par défaut [M+H]+ si inconnu

def calculate_molar_concentration(concentration: float, mz: float, adduct: str) -> float:
    """Calcule la concentration molaire à partir de la concentration en g/L et du m/z."""
    mass_difference = get_mass_difference(adduct)
    molecular_mass = mz - mass_difference  # masse moléculaire approximative
    return concentration / molecular_mass if molecular_mass > 0 else None

def load_target_compounds(compounds_file: Path) -> List[str]:
    """Charge la liste des composés cibles pour les calibrants."""
    df = pd.read_csv(compounds_file)
    return df['Compound'].tolist()

def load_calibration_samples(calibration_file: Path) -> pd.DataFrame:
    """Charge les données des échantillons de calibration."""
    df = pd.read_csv(calibration_file)
    concentrations = pd.DataFrame({
        'Sample': df['Name'],
        'conc': df['conc']  # Changé de conc_M à conc
    })
    return concentrations

def process_all_data(
    features_df: pd.DataFrame,
    feature_matrix_df: pd.DataFrame,
    calibration_df: pd.DataFrame
) -> pd.DataFrame:
    """Traite les données pour tous les échantillons en gardant l'intensité maximale."""
    results = []
    
    # On traite tous les composés de niveau 1
    level1_matches = features_df[features_df['confidence_level'] == 1].copy()
    
    for _, match_row in level1_matches.iterrows():
        compound = match_row['match_name']
        feature_id = f"{match_row['feature_id']}_mz{match_row['mz']:.4f}"
        samples = match_row['samples'].split(',')
        mz = match_row['mz']  # Récupération du m/z
        adduct = match_row['match_adduct']  # Récupération de l'adduit
        
        for sample in samples:
            sample = sample.strip()
            
            # On vérifie si c'est un échantillon de calibration
            is_calibration = sample in calibration_df['Sample'].values
            
            # Pour les composés dans les échantillons de calibration, on récupère la concentration
            conc_val = None
            conc_M = None
            if is_calibration:
                conc_val = calibration_df.loc[calibration_df['Sample'] == sample, 'conc'].iloc[0]
                if pd.notnull(conc_val):
                    conc_M = calculate_molar_concentration(conc_val, mz, adduct)
            
            intensity = feature_matrix_df.loc[sample, feature_id] if sample in feature_matrix_df.index and feature_id in feature_matrix_df.columns else None
            
            results.append({
                'Compound': compound,
                'SMILES': match_row['match_smiles'],
                'Feature_ID': feature_id,
                'Adduct': adduct,
                'RT': match_row['retention_time'],
                'DT': match_row['drift_time'],
                'CCS': match_row['CCS'],
                'Sample': sample,
                'Is_Calibration': is_calibration,
                'conc': conc_val,  # Concentration originale
                'conc_M': conc_M,  # Concentration molaire calculée
                'Intensity': intensity,
                'Confidence_Level': match_row['confidence_level'],
                'daphnia_LC50_48_hr_ug/L': match_row.get('daphnia_LC50_48_hr_ug/L'),
                'algae_EC50_72_hr_ug/L': match_row.get('algae_EC50_72_hr_ug/L'),
                'pimephales_LC50_96_hr_ug/L': match_row.get('pimephales_LC50_96_hr_ug/L'),
                'mz': mz  # Ajout du m/z pour référence
            })
    
    # Convertir en DataFrame
    results_df = pd.DataFrame(results)
    
    if results_df.empty:
        return pd.DataFrame()
    
    # Garder seulement l'intensité maximale pour chaque composé dans chaque échantillon
    max_intensity_df = results_df.loc[
        results_df.groupby(['Compound', 'Sample'])['Intensity'].idxmax()
    ]
    
    return max_intensity_df

def get_compound_summary(
    input_dir: Path,
    compounds_file: Path,
    calibration_file: Optional[Path] = None
) -> pd.DataFrame:
    """Génère un résumé des composés avec intensité maximale."""
    # Charger les données
    features_df = pd.read_parquet(input_dir / "feature_matrix/features_complete.parquet")
    
    # Lecture avec des limites augmentées
    table = pq.read_table(
        input_dir / "feature_matrix/feature_matrix.parquet",
        thrift_string_size_limit=1000*1024*1024,  # 1GB
        thrift_container_size_limit=1000*1024*1024
    )
    
    # Conversion en DataFrame
    feature_matrix_df = table.to_pandas()
    
    # Charger la liste des composés cibles et les échantillons de calibration
    target_compounds = load_target_compounds(compounds_file)
    calibration_df = load_calibration_samples(calibration_file) if calibration_file else None
    
    if calibration_df is None:
        return pd.DataFrame()
    
    # Traiter toutes les données
    results_df = process_all_data(
        features_df,
        feature_matrix_df,
        calibration_df
    )
    
    if results_df.empty:
        return pd.DataFrame()
    
    # Réorganiser les colonnes
    columns = [
        'Compound', 'Sample', 'Adduct',  # Colonnes prioritaires
        'SMILES', 'Feature_ID', 'RT', 'DT', 
        'CCS', 'Is_Calibration', 'conc', 'conc_M', 
        'Intensity', 'Confidence_Level', 'mz',
        'daphnia_LC50_48_hr_ug/L', 'algae_EC50_72_hr_ug/L', 
        'pimephales_LC50_96_hr_ug/L'
    ]
    
    return results_df[columns].sort_values(['Compound', 'Sample'])
