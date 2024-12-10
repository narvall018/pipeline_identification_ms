# scripts/quantification/calculate_rqmix.py

import pandas as pd
from pathlib import Path

def calculate_rqmix(sample_dir: Path, output_dir: Path) -> None:
    """
    Calcule les valeurs RQmix pour tous les échantillons et sauvegarde les résultats.
    
    Args:
        sample_dir: Dossier contenant les résultats de quantification
        output_dir: Dossier pour sauvegarder les résultats RQmix
    """
    # Création du dossier de sortie
    rqmix_dir = output_dir / "rqmix"
    rqmix_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialisation des résultats
    results = []
    
    # Traitement de chaque fichier de quantification
    for quant_file in sample_dir.glob("*_quantification.csv"):
        try:
            # Récupération du nom de l'échantillon
            sample_name = quant_file.stem.replace("_quantification", "")
            print(f"Traitement de {sample_name}...")
            
            # Lecture des données de quantification
            df = pd.read_csv(quant_file)
            
            # Vérification des colonnes nécessaires
            required_columns = [
                'identifier', 'conc',
                'daphnia_LC50_48_hr_ug/L',
                'algae_EC50_72_hr_ug/L',
                'pimephales_LC50_96_hr_ug/L'
            ]
            
            if not all(col in df.columns for col in required_columns):
                print(f"❌ Colonnes manquantes dans {quant_file.name}")
                continue
            
            # Conversion de la concentration de g/L en µg/L
            df['conc_ug_L'] = df['conc'] * 1e6
            
            # Calcul des RQ individuels avec gestion des valeurs invalides
            df['RQ_daphnia'] = df.apply(lambda row: 
                row['conc_ug_L'] / row['daphnia_LC50_48_hr_ug/L'] 
                if row['daphnia_LC50_48_hr_ug/L'] > 0 else 0, axis=1)
                
            df['RQ_algae'] = df.apply(lambda row: 
                row['conc_ug_L'] / row['algae_EC50_72_hr_ug/L']
                if row['algae_EC50_72_hr_ug/L'] > 0 else 0, axis=1)
                
            df['RQ_pimephales'] = df.apply(lambda row: 
                row['conc_ug_L'] / row['pimephales_LC50_96_hr_ug/L']
                if row['pimephales_LC50_96_hr_ug/L'] > 0 else 0, axis=1)
            
            # Remplacement des valeurs infinies par 0
            df = df.replace([float('inf'), -float('inf')], 0)
            
            # Calcul du RQmix pour chaque endpoint
            rqmix_result = {
                'Sample': sample_name,
                'RQmix_daphnia_LC50_48h': df['RQ_daphnia'].sum(),
                'RQmix_algae_EC50_72h': df['RQ_algae'].sum(),
                'RQmix_pimephales_LC50_96h': df['RQ_pimephales'].sum()
            }
            
            results.append(rqmix_result)
            
            # Sauvegarde des résultats détaillés pour cet échantillon
            detailed_df = df[[
                'identifier', 'conc', 'conc_ug_L',
                'daphnia_LC50_48_hr_ug/L', 'RQ_daphnia',
                'algae_EC50_72_hr_ug/L', 'RQ_algae',
                'pimephales_LC50_96_hr_ug/L', 'RQ_pimephales'
            ]].copy()
            
            detailed_df.to_csv(rqmix_dir / f"{sample_name}_rqmix_details.csv", index=False)
            print(f"✓ Résultats détaillés sauvegardés pour {sample_name}")
            
        except Exception as e:
            print(f"❌ Erreur lors du traitement de {quant_file.name}: {str(e)}")
    
    # Création et sauvegarde des résultats finaux
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(rqmix_dir / "rqmix_summary.csv", index=False)
        print(f"\n✓ Résultats RQmix sauvegardés dans {rqmix_dir}")
        
        # Affichage du résumé
        print("\nRésumé des RQmix:")
        print(results_df.to_string(index=False))
    else:
        print("❌ Aucun résultat généré")

