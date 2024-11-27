#main.py
#-*- coding:utf-8 -*-


# Importation des modules
import logging
import warnings
import pandas as pd
from pathlib import Path
from typing import Union
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
    plot_cluster_statistics
)

# Suppression des warnings pandas
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None


# Initialiser le logger
logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """
    Configure le système de logging pour enregistrer les événements dans un fichier.

    Args:
        None

    Returns:
        None
    """
    # Définition du répertoire où les logs seront enregistrés
    log_dir = Path("logs")

    # Crée le répertoire si nécessaire
    log_dir.mkdir(exist_ok=True)

    # Configure le système de logging
    logging.basicConfig(
        filename=log_dir / "peak_detection.log",  # Fichier où les logs seront écrits
        level=logging.INFO,                       # Niveau de log minimum (INFO et supérieur)
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Format des messages de log
        force=True                                # Remplace toute configuration de logging précédente
    )

    # Ajoute un message indiquant que le logging a été configuré
    logging.info("Logging configuré avec succès.")


def process_file(
    file_path: Union[str, Path],
    calibrator: CCSCalibrator,
    identifier: CompoundIdentifier,
    data_type: str = 'samples',
    total_files: int = 1,
    current_file: int = 1
) -> None:
    """
    Traite un fichier d'échantillon pour détecter les pics, calibrer les CCS, et identifier les composés.

    Args:
        file_path (Union[str, Path]): Chemin vers le fichier de l'échantillon à traiter.
        calibrator (CCSCalibrator): Instance du calibrateur CCS.
        identifier (CompoundIdentifier): Instance pour l'identification des composés.
        data_type (str): Type de données (par défaut 'samples').
        total_files (int): Nombre total de fichiers à traiter.
        current_file (int): Index du fichier en cours.

    Returns:
        None
    """
    try:
        # Récupère le nom de l'échantillon à partir du chemin du fichier
        sample_name = Path(file_path).stem

        # Affiche le titre du traitement pour l'échantillon courant
        print(f"\n{'=' * 80}")
        print(f"TRAITEMENT DE {sample_name} ({current_file}/{total_files})")
        print(f"{'=' * 80}")

        # Étape 1 : Lecture des données
        print("\n📊 Lecture des données...")
        data, metadata = read_parquet_data(file_path)
        print(f"   ✓ Données chargées : {len(data)} lignes")

        # Étape 2 : Préparation des données MS1
        print("\n🔍 Préparation des données MS1...")
        processed_data = prepare_data(data)
        if processed_data is None or processed_data.empty:
            print("   ✗ Aucune donnée MS1 valide")
            return
        print(f"   ✓ Données préparées : {len(processed_data)} lignes")

        # Étape 3 : Détection des pics
        print("\n🎯 Détection des pics...")
        peaks = detect_peaks(processed_data)
        if peaks.empty:
            print("   ✗ Aucun pic détecté")
            return
        print(f"   ✓ Pics détectés : {len(peaks)}")
        save_peaks(peaks, sample_name, "peaks", data_type, metadata)

        # Étape 4 : Clustering des pics
        print("\n🔄 Clustering des pics...")
        clustered_peaks = cluster_peaks(peaks)
        if clustered_peaks.empty:
            print("   ✗ Pas de pics après clustering")
            return
        print(f"   ✓ Pics après clustering : {len(clustered_peaks)}")
        save_peaks(clustered_peaks, sample_name, "clustered_peaks", data_type, metadata)

        # Étape 5 : Calibration des CCS
        print("\n🔵 Calibration CCS...")
        peaks_with_ccs = calibrator.calculate_ccs(clustered_peaks)
        if peaks_with_ccs.empty:
            print("   ✗ Erreur dans le calcul des CCS")
            return
        print(f"   ✓ CCS calculées pour {len(peaks_with_ccs)} pics")
        save_peaks(peaks_with_ccs, sample_name, "ccs_peaks", data_type, metadata)

        # Étape 6 : Identification des composés
        print("\n🔍 Identification des composés...")
        identification_dir = Path(f"data/intermediate/{data_type}/{sample_name}/ms1/identifications")
        matches_df = identifier.identify_compounds(peaks_with_ccs, identification_dir)

        if matches_df is None or matches_df.empty:
            print("   ✗ Aucun match trouvé")
            return
        print(f"   ✓ Matches trouvés : {len(matches_df)}")

        # Étape 7 : Extraction des données MS2
        matches_df = extract_ms2_for_matches(matches_df, file_path, identification_dir)

        # Résumé des résultats
        print(f"\n✨ Traitement terminé pour {sample_name}")
        print(f"   - Pics détectés : {len(peaks)}")
        print(f"   - Pics après clustering : {len(clustered_peaks)}")
        print(f"   - Pics avec CCS : {len(peaks_with_ccs)}")
        if matches_df is not None and not matches_df.empty:
            print(f"   - Matches avec MS2 : {len(matches_df)}")
        print(f"{'=' * 80}")

    except Exception as e:
        # Gestion des erreurs
        print(f"\n❌ Erreur lors du traitement de {sample_name}")
        logger.error(f"Erreur traitement {file_path} : {str(e)}")
        raise


def main() -> None:
    """
    Point d'entrée principal pour la pipeline d'analyse. Configure le logger,
    initialise les composants, traite les fichiers d'échantillons, et génère les visualisations.
    """
    # Configuration du système de logging
    setup_logging()
    print("\n🚀 DÉMARRAGE DE LA PIPELINE D'ANALYSE")
    print("=" * 80)

    try:
        # Étape 1 : Chargement des données de calibration CCS
        print("\n📈 Chargement des données de calibration CCS...")
        calibration_file = Path("data/input/calibration/CCS_calibration_data.csv")

        if not calibration_file.exists():
            raise FileNotFoundError(f"Fichier de calibration non trouvé : {calibration_file}")

        calibrator = CCSCalibrator(calibration_file)
        print("   ✓ Données de calibration chargées avec succès")

        # Étape 2 : Initialisation de l'identificateur
        print("\n📚 Initialisation de l'identification...")
        identifier = CompoundIdentifier()
        print("   ✓ Base de données chargée avec succès")

        # Étape 3 : Traitement des fichiers d'échantillons
        print("\n📁 Recherche des fichiers d'échantillons...")
        samples_dir = Path(Config.INPUT_SAMPLES)
        sample_files = list(samples_dir.glob("*.parquet"))

        if not sample_files:
            raise ValueError("Aucun fichier d'échantillon trouvé.")

        print(f"   ✓ {len(sample_files)} fichier(s) trouvé(s) pour traitement.")

        for idx, file_path in enumerate(sample_files, 1):
            # Traite chaque fichier d'échantillon individuellement
            process_file(file_path, calibrator, identifier, 'samples', len(sample_files), idx)

        # Étape 4 : Calcul des scores de similarité MS2 et niveaux de confiance
        print("\n📊 Calcul des scores de similarité MS2 et niveaux de confiance...")
        matches_paths = Path("data/intermediate/samples").glob("*/ms1/identifications/all_matches.parquet")
        
        for matches_path in matches_paths:
            print(f"   ℹ️ Traitement de {matches_path.parent.parent.parent.name}")
            add_ms2_scores(matches_path, identifier)
            print(f"   ✓ Scores MS2 et niveaux de confiance mis à jour")

        # Étape 5 : Génération de visualisations
        print("\n📊 Génération des visualisations...")
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        # Plot du nombre total de molécules par échantillon
        fig = plot_unique_molecules_per_sample("data/intermediate/samples")
        fig_path = output_dir / "molecules_per_sample.png"
        fig.savefig(fig_path)
        plt.close()

        # Plot des molécules niveau 1
        fig = plot_level1_molecules_per_sample("data/intermediate/samples")
        fig_path = output_dir / "level1_molecules_per_sample.png"
        fig.savefig(fig_path)
        plt.close()

        print("\n📊 Analyse des similarités entre échantillons...")
        
        # Heatmap toutes identifications
        fig_similarity = plot_sample_similarity_heatmap("data/intermediate/samples")
        fig_similarity.savefig(output_dir / "sample_similarity_heatmap_all.png")
        plt.close()

        # Heatmap niveau 1 uniquement
        fig_similarity_l1 = plot_sample_similarity_heatmap_by_confidence(
            "data/intermediate/samples", 
            confidence_levels=[1],
            title_suffix=" - Niveau 1"
        )
        fig_similarity_l1.savefig(output_dir / "sample_similarity_heatmap_level1.png")
        plt.close()

        # Heatmap niveaux 1+2
        fig_similarity_l12 = plot_sample_similarity_heatmap_by_confidence(
            "data/intermediate/samples", 
            confidence_levels=[1, 2],
            title_suffix=" - Niveaux 1 et 2"
        )
        fig_similarity_l12.savefig(output_dir / "sample_similarity_heatmap_level1_2.png")
        plt.close()

        # Heatmap niveaux 1+2+3
        fig_similarity_l123 = plot_sample_similarity_heatmap_by_confidence(
            "data/intermediate/samples", 
            confidence_levels=[1, 2, 3],
            title_suffix=" - Niveaux 1, 2 et 3"
        )
        fig_similarity_l123.savefig(output_dir / "sample_similarity_heatmap_level1_2_3.png")
        plt.close()

        # Analyse des clusters
        cluster_stats = analyze_sample_clusters("data/intermediate/samples", n_clusters=3)
        
        # Sauvegarde des statistiques de clusters dans un fichier texte
        with open(output_dir / "cluster_analysis.txt", "w") as f:
            f.write("Analyse des clusters d'échantillons\n")
            f.write("================================\n\n")
            for cluster_name, stats in cluster_stats.items():
                f.write(f"\n{cluster_name}:\n")
                f.write(f"Nombre d'échantillons: {stats['n_samples']}\n")
                f.write(f"Moyenne de molécules par échantillon: {stats['avg_molecules_per_sample']:.1f}\n")
                f.write("Molécules caractéristiques:\n")
                for molecule in stats['characteristic_molecules'][:10]:  # Top 10 molécules
                    f.write(f"- {molecule}\n")
                f.write("\nÉchantillons dans ce cluster:\n")
                for sample in stats['samples']:
                    f.write(f"- {sample}\n")
                f.write("\n" + "-"*50 + "\n")

        # Visualisation des statistiques des clusters
        fig_stats = plot_cluster_statistics(cluster_stats)
        fig_stats.savefig(output_dir / "cluster_statistics.png")
        plt.close()
        
        print(f"   ✓ Visualisations sauvegardées dans {output_dir}")

        # Fin du traitement
        print("\n✅ TRAITEMENT TERMINÉ AVEC SUCCÈS")
        print("=" * 80)

    except Exception as e:
        # Gestion des erreurs globales
        print("\n❌ ERREUR DANS LA PIPELINE")
        logger.error(f"Erreur pipeline : {str(e)}")
        raise


if __name__ == "__main__":
	main()
