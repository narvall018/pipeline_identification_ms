# main.py
from pathlib import Path
import logging
import warnings
import pandas as pd
from scripts.utils.io_handlers import read_parquet_data, save_peaks
from scripts.processing.peak_detection import prepare_data, detect_peaks, cluster_peaks
from scripts.processing.ccs_calibration import CCSCalibrator
from scripts.processing.identification import CompoundIdentifier
from scripts.config.config import Config
from scripts.processing.ms2_extraction import extract_ms2_for_matches
import subprocess
from scripts.processing.ms2_comparison_launcher import run_ms2_comparison
from scripts.processing.ms2_comparison import add_ms2_scores



logger = logging.getLogger(__name__)

# Suppression des warnings pandas
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

# Configuration du logger
logger = logging.getLogger(__name__)

def setup_logging():
    """Configuration du logging"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        filename=log_dir / "peak_detection.log",
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )

def process_file(file_path, calibrator, identifier, data_type='samples', total_files=1, current_file=1):
    """Traitement d'un fichier"""
    try:
        sample_name = Path(file_path).stem
        
        # ÉTAPE 1: DÉTECTION DES PICS
        print(f"\n{'='*80}")
        print(f"ÉTAPE 1/4 : DÉTECTION DES PICS - Fichier {current_file}/{total_files}")
        print(f"Traitement de: {sample_name}")
        print(f"{'='*80}")
        
        # 1. Lecture du fichier
        print("\n📊 Lecture des données...")
        data, metadata = read_parquet_data(file_path)
        print(f"   ✓ Données chargées: {len(data)} lignes")
        
        # 2. Préparation des données MS1
        print("\n🔍 Préparation des données MS1...")
        processed_data = prepare_data(data)
        if processed_data is not None:
            print(f"   ✓ Données préparées: {len(processed_data)} lignes")
        else:
            print("   ✗ Pas de données MS1 valides")
            return
        
        # 3. Détection des pics
        print("\n🎯 Détection des pics en cours...")
        peaks = detect_peaks(processed_data)
        if len(peaks) > 0:
            print(f"   ✓ Pics détectés: {len(peaks)}")
            save_peaks(peaks, sample_name, "peaks", data_type, metadata)
        else:
            print("   ✗ Aucun pic détecté")
            return
        
        # 4. Clustering des pics
        print("\n🔄 Clustering des pics...")
        clustered_peaks = cluster_peaks(peaks)
        if len(clustered_peaks) > 0:
            print(f"   ✓ Pics après clustering: {len(clustered_peaks)}")
            save_peaks(clustered_peaks, sample_name, "clustered_peaks", data_type, metadata)
        else:
            print("   ✗ Pas de pics après clustering")
            return
            
        # ÉTAPE 2: CALCUL DES CCS
        print(f"\n{'='*80}")
        print(f"ÉTAPE 2/4 : CALIBRATION ET CALCUL DES CCS - Fichier {current_file}/{total_files}")
        print(f"{'='*80}")
        
        print("\n🔵 Application de la calibration CCS...")
        peaks_with_ccs = calibrator.calculate_ccs(clustered_peaks)
        if len(peaks_with_ccs) > 0:
            print(f"   ✓ CCS calculées: {len(peaks_with_ccs)} pics")
            print(f"   ✓ Plage de CCS: {peaks_with_ccs['CCS'].min():.2f} - {peaks_with_ccs['CCS'].max():.2f} Å²")
            print(f"   ✓ CCS moyenne: {peaks_with_ccs['CCS'].mean():.2f} Å²")
            save_peaks(peaks_with_ccs, sample_name, "ccs_peaks", data_type, metadata)
        else:
            print("   ✗ Erreur dans le calcul des CCS")
            return

        # ÉTAPE 3: IDENTIFICATION DES COMPOSÉS
        print(f"\n{'='*80}")
        print(f"ÉTAPE 3/4 : IDENTIFICATION DES COMPOSÉS - Fichier {current_file}/{total_files}")
        print(f"{'='*80}")
        
        print("\n🔍 Recherche des composés dans la base de données...")
        identification_dir = Path(f"data/intermediate/{data_type}/{sample_name}/ms1/identifications")
        
        # Identification des composés
        matches_df = identifier.identify_compounds(
            peaks_with_ccs,
            identification_dir
        )
        
        if matches_df is not None and not matches_df.empty:
            n_total_matches = len(matches_df)
            n_unique_peaks = len(matches_df['peak_mz'].unique())
            print(f"   ✓ {n_unique_peaks} pics ont trouvé des matches")
            print(f"   ✓ {n_total_matches} matches totaux trouvés")
            print(f"   ✓ Moyenne de {n_total_matches/n_unique_peaks:.1f} matches par pic")
            
            # ÉTAPE 4: EXTRACTION MS2
            print(f"\n{'='*80}")
            print(f"ÉTAPE 4/4 : EXTRACTION MS2 - Fichier {current_file}/{total_files}")
            print(f"{'='*80}")
            
            # Extraction et mise à jour des matches avec MS2
            matches_df = extract_ms2_for_matches(matches_df, file_path, identification_dir)
            
        else:
            print("   ✗ Aucun match trouvé")   
       
        # Résumé final
        print(f"\n✨ Traitement complet pour {sample_name}")
        print(f"   - Pics initiaux: {len(peaks)}")
        print(f"   - Pics après clustering: {len(clustered_peaks)}")
        print(f"   - Pics avec CCS: {len(peaks_with_ccs)}")
        if matches_df is not None and not matches_df.empty:
            print(f"   - Pics identifiés: {n_unique_peaks}")
            n_with_ms2 = sum(len(mz_list) > 0 for mz_list in matches_df.get('peaks_mz_ms2', []))
            print(f"   - Matches avec MS2: {n_with_ms2}")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"\n❌ Erreur lors du traitement de {sample_name}")
        print(f"   {str(e)}")
        logger.error(f"Erreur traitement {file_path}: {str(e)}")
        raise

def main():
    setup_logging()
    print("\n🚀 DÉMARRAGE DE LA PIPELINE D'ANALYSE")
    print("=" * 80)
    
    try:
        # Initialisation du calibrateur CCS
        print("\n📈 Chargement des données de calibration CCS...")
        calibration_file = Path("data/input/calibration/CCS_calibration_data.csv")
        
        if not calibration_file.exists():
            raise FileNotFoundError(f"Fichier de calibration non trouvé: {calibration_file}")
            
        calibrator = CCSCalibrator(calibration_file)
        print("   ✓ Données de calibration chargées avec succès")
        
        # Initialisation de l'identificateur
        print("\n📚 Initialisation de l'identification...")
        identifier = CompoundIdentifier()
        print("   ✓ Base de données chargée avec succès")
        
        # Traitement des échantillons
        samples_dir = Path(Config.INPUT_SAMPLES)
        sample_files = list(samples_dir.glob("*.parquet"))
        if sample_files:
            print(f"\n📁 Traitement des échantillons ({len(sample_files)} fichiers)")
            for idx, file_path in enumerate(sample_files, 1):
                process_file(file_path, calibrator, identifier, 'samples', len(sample_files), idx)
 
            
            # Après le traitement de tous les fichiers, faire la comparaison MS2
            print("\n📊 Calcul des scores de similarité MS2...")
            for sample_path in Path("data/intermediate/samples").glob("*/ms1/identifications/all_matches.parquet"):
                add_ms2_scores(sample_path, identifier)
 
 
        print("\n📊 Génération des visualisations...")
        from scripts.visualization.plotting import plot_unique_molecules_per_sample
        
        # Créer le dossier output s'il n'existe pas
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Générer et sauvegarder le plot
        fig = plot_unique_molecules_per_sample("data/intermediate/samples")
        fig.savefig(output_dir / "molecules_per_sample.png")
        print("   ✓ Visualisation sauvegardée dans output/molecules_per_sample.png")
        
        print("\n✅ TRAITEMENT TERMINÉ AVEC SUCCÈS")
        print("=" * 80)
        
    except Exception as e:
        print("\n❌ ERREUR DANS LA PIPELINE")
        print(f"   {str(e)}")
        logger.error(f"Erreur pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
