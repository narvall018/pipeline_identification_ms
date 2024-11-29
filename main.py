#main.py
#-*- coding:utf-8 -*-

# -*- coding:utf-8 -*-

import numpy as np
import logging
import warnings
import pandas as pd
from pathlib import Path
from typing import Union, List
from scripts.config.config import Config
from scripts.utils.io_handlers import read_parquet_data, save_peaks
from scripts.processing.peak_detection import prepare_data, detect_peaks, cluster_peaks
from scripts.processing.ccs_calibration import CCSCalibrator
from scripts.processing.identification import CompoundIdentifier
from scripts.processing.ms2_extraction import extract_ms2_for_matches
from scripts.processing.ms2_comparaison import add_ms2_scores
import matplotlib.pyplot as plt 
from scripts.visualization.plotting import (
    plot_unique_molecules_per_sample,
    plot_level1_molecules_per_sample,
    plot_sample_similarity_heatmap,
    plot_sample_similarity_heatmap_by_confidence,
    analyze_sample_clusters,
    plot_cluster_statistics,
    plot_level1_molecule_distribution_bubble
)
from scripts.visualization.plotting import plot_tics_interactive
from scripts.utils.replicate_handling import group_replicates
from scripts.processing.replicate_processing import process_sample_with_replicates
from scripts.processing.blank_processing import (
    process_blank_with_replicates,
    subtract_blank_peaks
)

# Suppression des warnings pandas
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

# Initialiser le logger
logger = logging.getLogger(__name__)

def setup_logging() -> None:
    """Configure le système de logging pour enregistrer les événements dans un fichier."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        filename=log_dir / "peak_detection.log",
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )
    logging.info("Logging configuré avec succès.")

def process_blank_files(blank_files: List[Path]) -> pd.DataFrame:
    """Traite tous les fichiers blanks et retourne les pics combinés."""
    blank_peaks = pd.DataFrame()
    
    if blank_files:
        print(f"   ✓ {len(blank_files)} fichier(s) blank trouvé(s):")
        for blank_file in blank_files:
            print(f"      - {blank_file.name}")
            
        for blank_file in blank_files:
            blank_peaks_group = process_blank_with_replicates(
                blank_file.stem,
                [blank_file],
                Path("data/intermediate/blanks")
            )
            if not blank_peaks_group.empty:
                blank_peaks = pd.concat([blank_peaks, blank_peaks_group])
        print(f"   ✓ {len(blank_peaks)} pics de blank détectés")
    else:
        print("   ℹ️ Aucun blank trouvé dans data/input/blanks/")
        
    return blank_peaks

def process_full_sample(
    base_name: str,
    replicates: List[Path],
    blank_peaks: pd.DataFrame,
    calibrator: CCSCalibrator,
    identifier: CompoundIdentifier,
    output_base_dir: Path
) -> None:
    """
    Traite un échantillon complet avec l'ordre correct des étapes:
    1. Traitement des réplicats
    2. Soustraction du blank
    3. Calibration CCS
    4. Identification
    5. MS2
    """
    print(f"\n{'='*80}")
    print(f"TRAITEMENT DE {base_name} ({len(replicates)} réplicats)")
    print(f"{'='*80}")
    
    # Création des dossiers de sortie
    output_dir = output_base_dir / base_name / "ms1"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Traitement des réplicats et obtention des pics communs
    common_peaks = process_sample_with_replicates(
        base_name,
        replicates,
        Path("data/intermediate/samples")
    )
    
    if common_peaks.empty:
        print(f"   ✗ Pas de pics trouvés pour {base_name}")
        return
    
    # 2. Soustraction du blank
    if not blank_peaks.empty:
        clean_peaks = subtract_blank_peaks(common_peaks, blank_peaks)
        pics_retires = len(common_peaks) - len(clean_peaks)
        pourcentage = (pics_retires / len(common_peaks) * 100) if len(common_peaks) > 0 else 0
        print(f"   ✓ {len(clean_peaks)} pics après soustraction du blank")
    else:
        clean_peaks = common_peaks
        
    if clean_peaks.empty:
        print(f"   ✗ Pas de pics après soustraction du blank pour {base_name}")
        # Sauvegarde un DataFrame vide
        output_file = output_dir / "common_peaks.parquet"
        pd.DataFrame().to_parquet(output_file)
        return
        
    # 3. Calibration CCS sur les pics nettoyés
    print("\n🔵 Calibration CCS...")
    peaks_with_ccs = calibrator.calculate_ccs(clean_peaks)
    if peaks_with_ccs.empty:
        print(f"   ✗ Erreur dans le calcul des CCS")
        return
        
    print(f"   ✓ CCS calculées pour {len(peaks_with_ccs)} pics")
    print(f"   ✓ Plage de CCS: {peaks_with_ccs['CCS'].min():.2f} - {peaks_with_ccs['CCS'].max():.2f} Å²")
    print(f"   ✓ CCS moyenne: {peaks_with_ccs['CCS'].mean():.2f} Å²")
    
    # Sauvegarde des pics finaux avec CCS
    output_file = output_dir / "common_peaks.parquet"
    peaks_with_ccs.to_parquet(output_file)
    print(f"   ✓ Pics finaux sauvegardés dans {output_file}")
    
    # 4. Identification uniquement si on a des pics avec CCS
    if not peaks_with_ccs.empty:
        print("\n🔍 Identification des composés...")
        identification_dir = output_dir / "identifications"
        matches_df = identifier.identify_compounds(peaks_with_ccs, identification_dir)
        
        # 5. MS2 uniquement si on a des identifications
        if matches_df is not None and not matches_df.empty:
            data, _ = read_parquet_data(replicates[0])
            ms_levels = data['mslevel'].value_counts()
            
            matches_df = extract_ms2_for_matches(
                matches_df,
                replicates[0],
                identification_dir
            )
            
            if matches_df is not None:
                if 'peaks_intensities_ms2' in matches_df.columns:
                    ms2_count = sum(
                        len(peaks) > 0 
                        for peaks in matches_df['peaks_intensities_ms2']
                        if isinstance(peaks, (list, np.ndarray))
                    )
                    total_matches = len(matches_df)



def generate_visualizations(output_dir: Path):
    """Génère toutes les visualisations de la pipeline."""
    output_dir.mkdir(exist_ok=True)

    # Plot du nombre total de molécules par échantillon
    fig = plot_unique_molecules_per_sample("data/intermediate/samples")
    fig.savefig(output_dir / "molecules_per_sample.png")
    plt.close()

    # Plot des molécules niveau 1
    fig = plot_level1_molecules_per_sample("data/intermediate/samples")
    fig.savefig(output_dir / "level1_molecules_per_sample.png")
    plt.close()

    # Bubble plot niveau 1
    fig_bubble = plot_level1_molecule_distribution_bubble("data/intermediate/samples") 
    fig_bubble.savefig(output_dir / "level1_molecule_distribution_bubble.png",
                      bbox_inches='tight',
                      dpi=300)
    plt.close()

    # TIC 
    plot_tics_interactive(Config.INPUT_SAMPLES, output_dir)

    # Heatmaps
    fig_similarity = plot_sample_similarity_heatmap("data/intermediate/samples")
    fig_similarity.savefig(output_dir / "sample_similarity_heatmap_all.png")
    plt.close()

    fig_similarity_l1 = plot_sample_similarity_heatmap_by_confidence(
        "data/intermediate/samples", 
        confidence_levels=[1],
        title_suffix=" - Niveau 1"
    )
    fig_similarity_l1.savefig(output_dir / "sample_similarity_heatmap_level1.png")
    plt.close()

    fig_similarity_l12 = plot_sample_similarity_heatmap_by_confidence(
        "data/intermediate/samples", 
        confidence_levels=[1, 2],
        title_suffix=" - Niveaux 1 et 2"
    )
    fig_similarity_l12.savefig(output_dir / "sample_similarity_heatmap_level1_2.png")
    plt.close()

    fig_similarity_l123 = plot_sample_similarity_heatmap_by_confidence(
        "data/intermediate/samples", 
        confidence_levels=[1, 2, 3],
        title_suffix=" - Niveaux 1, 2 et 3"
    )
    fig_similarity_l123.savefig(output_dir / "sample_similarity_heatmap_level1_2_3.png")
    plt.close()

def analyze_and_save_clusters(output_dir: Path):
    """Analyse et sauvegarde les statistiques des clusters."""
    cluster_stats = analyze_sample_clusters("data/intermediate/samples", n_clusters=3)
    
    with open(output_dir / "cluster_analysis.txt", "w") as f:
        f.write("Analyse des clusters d'échantillons\n")
        f.write("================================\n\n")
        for cluster_name, stats in cluster_stats.items():
            f.write(f"\n{cluster_name}:\n")
            f.write(f"Nombre d'échantillons: {stats['n_samples']}\n")
            f.write(f"Moyenne de molécules par échantillon: {stats['avg_molecules_per_sample']:.1f}\n")
            f.write("Molécules caractéristiques:\n")
            for molecule in stats['characteristic_molecules'][:10]:
                f.write(f"- {molecule}\n")
            f.write("\nÉchantillons dans ce cluster:\n")
            for sample in stats['samples']:
                f.write(f"- {sample}\n")
            f.write("\n" + "-"*50 + "\n")

    fig_stats = plot_cluster_statistics(cluster_stats)
    fig_stats.savefig(output_dir / "cluster_statistics.png")
    plt.close()

def main() -> None:
    """Point d'entrée principal de la pipeline d'analyse."""
    # Configuration du système de logging
    setup_logging()
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

        # 3. Traitement des blanks
        print("\n📁 Recherche des blanks...")
        blank_dir = Path("data/input/blanks")
        blank_files = list(blank_dir.glob("*.parquet"))
        blank_peaks = process_blank_files(blank_files)

        # 4. Traitement des échantillons
        print("\n📁 Recherche des fichiers d'échantillons...")
        samples_dir = Path(Config.INPUT_SAMPLES)
        sample_files = list(samples_dir.glob("*.parquet"))

        if not sample_files:
            raise ValueError("Aucun fichier d'échantillon trouvé.")
            
        replicate_groups = group_replicates(sample_files)
        
        print(f"   ✓ {len(replicate_groups)} échantillons trouvés:")
        for base_name, replicates in replicate_groups.items():
            print(f"      - {base_name}: {len(replicates)} réplicat(s)")

        # Traitement de chaque échantillon
        for base_name, replicates in replicate_groups.items():
            process_full_sample(
                base_name,
                replicates,
                blank_peaks,
                calibrator,
                identifier,
                Path("data/intermediate/samples")
            )

        # 5. Calcul des scores MS2
        print("\n📊 Calcul des scores de similarité MS2 et niveaux de confiance...")
        matches_paths = Path("data/intermediate/samples").glob("*/ms1/identifications/all_matches.parquet")
        for matches_path in matches_paths:
            print(f"   ℹ️ Traitement de {matches_path.parent.parent.parent.name}")
            add_ms2_scores(matches_path, identifier)
            print(f"   ✓ Scores MS2 et niveaux de confiance mis à jour")

        # 6. Visualisations
        print("\n📊 Génération des visualisations...")
        output_dir = Path("output")
        generate_visualizations(output_dir)
        
        print("\n📊 Analyse des similarités entre échantillons...")
        analyze_and_save_clusters(output_dir)
        
        print(f"   ✓ Visualisations sauvegardées dans {output_dir}")

        # Fin du traitement
        print("\n✅ TRAITEMENT TERMINÉ AVEC SUCCÈS")
        print("=" * 80)

    except Exception as e:
        print("\n❌ ERREUR DANS LA PIPELINE")
        logger.error(f"Erreur pipeline : {str(e)}")
        raise


if __name__ == "__main__":
	main()
