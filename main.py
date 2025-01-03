#main.py
#-*- coding:utf-8 -*--

import gc 
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import logging
import warnings
import pandas as pd
from pathlib import Path
from typing import Union, List, Dict, Tuple
from queue import Queue
from io import StringIO
import sys
from contextlib import redirect_stdout
import time
from tqdm import tqdm
import matplotlib.pyplot as plt 
from scripts.config.config import Config
from scripts.utils.io_handlers import read_parquet_data, save_peaks
from scripts.processing.peak_detection import prepare_data, detect_peaks, cluster_peaks
from scripts.processing.ccs_calibration import CCSCalibrator
from scripts.processing.identification import CompoundIdentifier
from scripts.visualization.plotting import (
    plot_unique_molecules_per_sample,
    plot_level1_molecules_per_sample,
    plot_sample_similarity_heatmap,
    plot_sample_similarity_heatmap_by_confidence,
    plot_level1_molecule_distribution_bubble,
    analyze_sample_clusters,
    plot_cluster_statistics,
    analyze_and_save_clusters,
    plot_tics_interactive,
    analyze_categories
)
from scripts.utils.replicate_handling import group_replicates
from scripts.processing.replicate_processing import process_sample_with_replicates
from scripts.processing.blank_processing import (
    process_blank_with_replicates,
    subtract_blank_peaks
)
from scripts.processing.feature_matrix import create_feature_matrix



# Suppression des warnings pandas
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None


# Initialiser le logger
logger = logging.getLogger(__name__)

def setup_logging() -> None:
    """Configure le système de logging."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        filename=log_dir / "peak_detection.log",
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )

class CaptureOutput:
    """Capture la sortie standard pour chaque processus."""
    def __init__(self):
        self.output = StringIO()
        self.stdout = sys.stdout
        
    def __enter__(self):
        sys.stdout = self.output
        return self.output
        
    def __exit__(self, *args):
        sys.stdout = self.stdout

class SampleResult:
    """Stocke les résultats du traitement d'un échantillon."""
    def __init__(self, name: str, peaks_df: pd.DataFrame, processing_time: float, logs: str):
        self.name = name
        self.peaks_df = peaks_df
        self.processing_time = processing_time
        self.logs = logs
        self.success = not peaks_df.empty
        
        # Extraire les statistiques des logs
        self.initial_peaks = 0
        self.after_clustering = 0
        self.after_blank = len(peaks_df) if not peaks_df.empty else 0  # Nombre de pics finaux
        self.final_peaks = len(peaks_df) if not peaks_df.empty else 0
        
        for line in logs.split('\n'):
            if "Pics initiaux:" in line:
                try:
                    self.initial_peaks = int(line.split(": ")[1])
                except:
                    pass
            elif "après clustering:" in line and "après clustering: {len" not in line:
                try:
                    self.after_clustering = int(line.split(": ")[1])
                except:
                    pass
            elif "Pics après soustraction" in line:
                try:
                    self.after_blank = int(line.split(": ")[1])
                except:
                    pass


def process_blank_files(blank_files: List[Path]) -> pd.DataFrame:
    """Traite tous les fichiers blanks."""
    blank_peaks = pd.DataFrame()
    
    # Message concernant la présence ou l'absence de blanks
    if blank_files:
        print(f"   ✓ {len(blank_files)} fichier(s) blank trouvé(s):")
        for blank_file in blank_files:
            print(f"      - {blank_file.name}")
    else:
        print("   ℹ️ Aucun blank trouvé dans data/input/blanks/")
            
    if blank_files:
        for blank_file in blank_files:
            blank_peaks_group = process_blank_with_replicates(
                blank_file.stem,
                [blank_file],
                Path("data/intermediate/blanks")
            )
            if not blank_peaks_group.empty:
                blank_peaks = pd.concat([blank_peaks, blank_peaks_group])
        if not blank_peaks.empty:
            print(f"   ✓ {len(blank_peaks)} pics de blank détectés")
            
    return blank_peaks

def generate_molecules_per_sample(output_dir: Path):
    fig = plot_unique_molecules_per_sample(output_dir)
    fig.savefig(output_dir / "molecules_per_sample.png", bbox_inches='tight', dpi=300)
    plt.close(fig)

def generate_level1_molecules(output_dir: Path):
    fig = plot_level1_molecules_per_sample(output_dir)
    fig.savefig(output_dir / "level1_molecules_per_sample.png", bbox_inches='tight', dpi=300)
    plt.close(fig)

def generate_bubble_plot(output_dir: Path):
    fig = plot_level1_molecule_distribution_bubble(output_dir)
    fig.savefig(output_dir / "level1_molecule_distribution_bubble.png", bbox_inches='tight', dpi=300)
    plt.close(fig)

def generate_similarity_heatmap(output_dir: Path):
    fig = plot_sample_similarity_heatmap(output_dir)
    fig.savefig(output_dir / "sample_similarity_heatmap_all.png", bbox_inches='tight', dpi=300)
    plt.close(fig)

def generate_level1_heatmap(output_dir: Path):
    fig = plot_sample_similarity_heatmap_by_confidence(
        output_dir, 
        confidence_levels=[1],
        title_suffix=" - Niveau 1"
    )
    fig.savefig(output_dir / "sample_similarity_heatmap_level1.png", bbox_inches='tight', dpi=300)
    plt.close(fig)

def generate_level12_heatmap(output_dir: Path):
    fig = plot_sample_similarity_heatmap_by_confidence(
        output_dir, 
        confidence_levels=[1, 2],
        title_suffix=" - Niveaux 1 et 2"
    )
    fig.savefig(output_dir / "sample_similarity_heatmap_level12.png", bbox_inches='tight', dpi=300)
    plt.close(fig)

def generate_level123_heatmap(output_dir: Path):
    fig = plot_sample_similarity_heatmap_by_confidence(
        output_dir, 
        confidence_levels=[1, 2, 3],
        title_suffix=" - Niveaux 1, 2 et 3"
    )
    fig.savefig(output_dir / "sample_similarity_heatmap_level123.png", bbox_inches='tight', dpi=300)
    plt.close(fig)

def generate_tics(output_dir: Path):
    plot_tics_interactive(Path("data/input/samples"), output_dir)


def generate_visualizations(output_dir: Path) -> None:
    """Génère toutes les visualisations de la pipeline en parallèle."""
    try:
        print("\n📊 Génération des visualisations...")
        output_dir.mkdir(exist_ok=True)
        
        # Vérifier l'existence du fichier d'identifications dans le sous-dossier feature_matrix
        identifications_file = output_dir / "feature_matrix" / "features_complete.parquet"
        if not identifications_file.exists():
            raise FileNotFoundError(f"Fichier d'identifications non trouvé: {identifications_file}")

        # Liste des tâches à exécuter avec leurs arguments
        tasks = [
            (generate_molecules_per_sample, output_dir),
            (generate_level1_molecules, output_dir),
            (generate_bubble_plot, output_dir),
            (generate_similarity_heatmap, output_dir),
            (generate_level1_heatmap, output_dir),
            (generate_level12_heatmap, output_dir),
            (generate_level123_heatmap, output_dir)
            #,(generate_tics, output_dir)
        ]

        # Exécution parallèle des tâches
        max_workers = min(mp.cpu_count(), len(tasks))

        futures_to_task = {}
        with ProcessPoolExecutor(max_workers=2) as executor:
            # Soumettre toutes les tâches
            for func, arg in tasks:
                future = executor.submit(func, arg)
                futures_to_task[future] = func.__name__
            
            # Attendre leur complétion avec une barre de progression
            with tqdm(total=len(futures_to_task), desc="Génération des visualisations") as pbar:
                for future in as_completed(futures_to_task):
                    task_name = futures_to_task[future]
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Erreur lors de la génération de {task_name}: {str(e)}")
                    pbar.update(1)

        print(f"   ✓ Visualisations sauvegardées dans {output_dir}")

    except Exception as e:
        print(f"❌ Erreur lors de la création des visualisations: {str(e)}")
        raise


def process_single_sample(
    args: Tuple[str, List[Path], pd.DataFrame, CCSCalibrator, Path]
) -> Tuple[str, SampleResult]:
    """Traite un seul échantillon en capturant sa sortie."""
    base_name, replicates, blank_peaks, calibrator, output_base_dir = args
    start_time = time.time()
    
    with CaptureOutput() as output:
        try:
            # 1. Traitement des réplicats 📊
            common_peaks = process_sample_with_replicates(
                base_name,
                replicates,
                output_base_dir
            )
            
            if common_peaks.empty:
                print(f"✗ Pas de pics trouvés pour {base_name}")
                return base_name, SampleResult(
                    base_name, pd.DataFrame(), 
                    time.time() - start_time, 
                    output.getvalue()
                )
            
            # 2. Soustraction du blank 
            if not blank_peaks.empty:
                clean_peaks = subtract_blank_peaks(common_peaks, blank_peaks)
            else:
                clean_peaks = common_peaks
                
            if clean_peaks.empty:
                print(f"✗ Pas de pics après soustraction du blank pour {base_name}")
                return base_name, SampleResult(
                    base_name, pd.DataFrame(), 
                    time.time() - start_time, 
                    output.getvalue()
                )
            
            # 3. Calibration CCS 
            peaks_with_ccs = calibrator.calculate_ccs(clean_peaks)
            
            # Mise à jour des statistiques
            peaks_stats = {
                'initial': len(common_peaks),
                'after_clustering': len(clean_peaks),  # Correspond aux pics après clustering
                'after_blank': len(clean_peaks),  # Nombre identique aux pics finaux
                'final': len(peaks_with_ccs)
            }
            
            if not peaks_with_ccs.empty:
                print(f"✓ CCS calculées pour {len(peaks_with_ccs)} pics")
                print(f"✓ Plage de CCS: {peaks_with_ccs['CCS'].min():.2f} - {peaks_with_ccs['CCS'].max():.2f} Å²")
                print(f"✓ CCS moyenne: {peaks_with_ccs['CCS'].mean():.2f} Å²")
                
                # Sauvegarde
                output_dir = output_base_dir / base_name / "ms1"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / "common_peaks.parquet"
                peaks_with_ccs.to_parquet(output_file)
                
            return base_name, SampleResult(
                base_name, peaks_with_ccs,
                time.time() - start_time,
                output.getvalue()
            )
            
        except Exception as e:
            print(f"❌ Erreur lors du traitement de {base_name}: {str(e)}")
            return base_name, SampleResult(
                base_name, pd.DataFrame(),
                time.time() - start_time,
                output.getvalue()
            )

def process_samples_parallel(
    replicate_groups: Dict[str, List[Path]],
    blank_peaks: pd.DataFrame,
    calibrator: CCSCalibrator,
    output_base_dir: Path,
    max_workers: int = None
) -> Dict[str, SampleResult]:
    """Traite tous les échantillons en parallèle avec gestion de la mémoire."""
    if max_workers is None:
        max_workers = min(2, mp.cpu_count())
    
    total_samples = len(replicate_groups)    
    print("\n" + "="*80)
    print(f"TRAITEMENT DES ÉCHANTILLONS ({total_samples} échantillons)")
    print("="*80)
    
    total_start_time = time.time()
    results = {}
    
    # Traiter les échantillons par petits groupes
    batch_size = 2  # Réduit à 2 échantillons à la fois
    sample_items = list(replicate_groups.items())
    
    with tqdm(total=len(sample_items), unit="échantillon") as pbar:
        for i in range(0, len(sample_items), batch_size):
            batch_items = sample_items[i:i + batch_size]
            
            process_args = [
                (base_name, replicates, blank_peaks, calibrator, output_base_dir)
                for base_name, replicates in batch_items
            ]
            
            try:
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    future_to_name = {
                        executor.submit(process_single_sample, args): args[0]
                        for args in process_args
                    }
                    
                    for future in as_completed(future_to_name):
                        base_name = future_to_name[future]
                        try:
                            _, result = future.result()
                            results[base_name] = result
                        except Exception as e:
                            print(f"\n❌ Erreur pour {base_name}: {str(e)}")
                            # Créer un SampleResult vide pour les échantillons en erreur
                            results[base_name] = SampleResult(
                                base_name, 
                                pd.DataFrame(), 
                                0.0,
                                f"Erreur: {str(e)}"
                            )
                        finally:
                            pbar.update(1)
                
            except Exception as e:
                print(f"\n❌ Erreur dans le batch {i//batch_size + 1}: {str(e)}")
                # Gérer les échantillons non traités dans ce batch
                for name, _ in batch_items:
                    if name not in results:
                        results[name] = SampleResult(
                            name,
                            pd.DataFrame(),
                            0.0,
                            f"Erreur batch: {str(e)}"
                        )
                        pbar.update(1)
            
            # Force la libération de la mémoire
            gc.collect()
            
            # Sauvegarde intermédiaire des statistiques
            stats_df = pd.DataFrame([{
                'Échantillon': r.name,
                'Temps (s)': r.processing_time,
                'Pics initiaux': r.initial_peaks,
                'Pics après clustering': r.after_clustering,
                'Pics après blank': r.after_blank,
                'Pics finaux': r.final_peaks,
                'Statut': 'Succès' if r.success else 'Échec'
            } for r in results.values()])
            
            stats_file = output_base_dir / "processing_statistics.csv"
            stats_df.to_csv(stats_file, index=False)
    
    total_time = time.time() - total_start_time
    success_count = sum(1 for r in results.values() if r.success)
    failed_count = len(results) - success_count
    
    print("\n" + "="*80)
    print("RÉCAPITULATIF")
    print("="*80)
    
    print(f"\nTemps total: {total_time:.2f} secondes")
    print(f"Échantillons traités: {success_count}/{len(results)}")
    if failed_count > 0:
        print(f"❌ Échantillons en échec: {failed_count}")
    
    print("\nDétails par échantillon:")
    for name, result in results.items():
        status = "✓" if result.success else "✗"
        print(f"\n{status} {name}")
        print(f"  • Temps de traitement: {result.processing_time:.2f}s")
        print(f"  • Pics initiaux: {result.initial_peaks}")
        if result.success:
            print(f"  • Pics après clustering: {result.after_clustering}")
            print(f"  • Pics après blank: {result.after_blank}")
            print(f"  • Pics finaux: {result.final_peaks}")
            peaks_df = result.peaks_df
            if not peaks_df.empty and 'CCS' in peaks_df.columns:
                ccs_stats = peaks_df['CCS'].agg(['min', 'max', 'mean'])
                print(f"  • CCS min-max: {ccs_stats['min']:.2f} - {ccs_stats['max']:.2f} Å²")
                print(f"  • CCS moyenne: {ccs_stats['mean']:.2f} Å²")
        else:
            print(f"  • Erreur: {result.logs}")
    
    # Sauvegarde finale des statistiques
    final_stats_df = pd.DataFrame([{
        'Échantillon': r.name,
        'Temps (s)': r.processing_time,
        'Pics initiaux': r.initial_peaks,
        'Pics après clustering': r.after_clustering,
        'Pics après blank': r.after_blank,
        'Pics finaux': r.final_peaks,
        'Statut': 'Succès' if r.success else 'Échec'
    } for r in results.values()])
    
    stats_file = output_base_dir / "processing_statistics.csv"
    final_stats_df.to_csv(stats_file, index=False)
    print(f"\nStatistiques sauvegardées dans {stats_file}")
    
    return results


def main() -> None:
    """Point d'entrée principal de la pipeline."""
    setup_logging()
    start_time = time.time()
    print("\n🚀 DÉMARRAGE DE LA PIPELINE D'ANALYSE")
    print("=" * 80)

    try:
        # 1. Chargement des données de calibration CCS
        print("\n📈 Chargement des données de calibration CCS...")
        calibration_file = Path("data/input/calibration/CCS_calibration_data.csv")
        if not calibration_file.exists():
            raise FileNotFoundError(f"Fichier de calibration non trouvé : {calibration_file}")
        calibrator = CCSCalibrator(calibration_file)
        print("   ✓ Données de calibration chargées avec succès")

        # 2. Initialisation de l'identificateur
        print("\n📚 Initialisation de l'identification...")
        identifier = CompoundIdentifier()
        print("   ✓ Base de données chargée avec succès")

        # 3. Recherche des fichiers
        print("\n📁 Recherche des blanks...")
        blank_dir = Path("data/input/blanks")
        blank_files = list(blank_dir.glob("*.parquet"))
        if blank_files:
            print(f"   ✓ {len(blank_files)} fichier(s) blank trouvé(s):")
            for blank_file in blank_files:
                print(f"      - {blank_file.name}")
        else:
            print("   ℹ️ Aucun blank trouvé dans data/input/blanks/")

        print("\n📁 Recherche des échantillons...")
        samples_dir = Path(Config.INPUT_SAMPLES)
        sample_files = list(samples_dir.glob("*.parquet"))

        if not sample_files:
            raise ValueError("Aucun fichier d'échantillon trouvé.")
            
        replicate_groups = group_replicates(sample_files)
        print(f"   ✓ {len(replicate_groups)} échantillons trouvés:")
        for base_name, replicates in replicate_groups.items():
            print(f"      - {base_name}: {len(replicates)} réplicat(s)")

        print("\n" + "="*80)
        print("TRAITEMENT DES ÉCHANTILLONS")
        print("=" * 80)

        # 4. Traitement des blanks
        if blank_files:
            blank_peaks = process_blank_with_replicates(
                blank_files[0].stem,
                blank_files,
                Path("data/intermediate/blanks")
            )
        else:
            blank_peaks = pd.DataFrame()

        # 5. Traitement parallèle des échantillons
        results = process_samples_parallel(
            replicate_groups,
            blank_peaks,
            calibrator,
            Path("data/intermediate/samples")
        )
        
        # Vérifier si des échantillons ont été traités avec succès
        if not any(r.success for r in results.values()):
            raise Exception("Aucun échantillon n'a été traité avec succès")

        print("\n" + "="*80)
        print("ALIGNEMENT DES FEATURES")
        print("="*80)
        
        # 6. Feature Matrix et identification
        print("\n📊 Création de la matrice des features...")
        create_feature_matrix(
            input_dir=Path("data/intermediate/samples"),
            output_dir=Path("output/feature_matrix"),
            identifier=identifier
        )

        print("\n" + "="*80)
        print("GÉNÉRATION DES VISUALISATIONS")
        print("="*80)
        
        # 7. Visualisations
        output_dir = Path("output")
        generate_visualizations(output_dir)
        
        print("\n" + "="*80)
        print("ANALYSE DES SIMILARITÉS")
        print("="*80)
        
        print("\n📊 Analyse des clusters d'échantillons...")
        analyze_and_save_clusters(output_dir)

        print("\n" + "="*80)
        print("ANALYSE DES CATÉGORIES")
        print("="*80)
    
        print("\n📊 Analyse des catégories de molécules...")
        analyze_categories(output_dir)

        total_time = time.time() - start_time
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)
        

        print("\n" + "="*80)
        print(" ✅ FIN DU TRAITEMENT")
        print("="*80)
        print("\n Pipeline d'analyse terminée avec succès")
        if minutes > 0:
            print(f"   • Temps de calcul total: {minutes} min {seconds} sec")
        else:
            print(f"   • Temps de calcul total: {seconds} sec")
        print(f"   • {len(replicate_groups)} échantillons traités")
        print("=" * 80)

    except Exception as e:
        print("\n❌ ERREUR DANS LA PIPELINE")
        logger.error(f"Erreur pipeline : {str(e)}")
        raise

if __name__ == "__main__":
    main()

