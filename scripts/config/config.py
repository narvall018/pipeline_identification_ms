#scripts/config/config.py
#-*- coding:utf-8 -*-

from pathlib import Path
from typing import Dict, Union
from dataclasses import dataclass, field

@dataclass
class PathConfig:
    """Configuration des chemins du projet"""
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    INPUT_DIR: Path = DATA_DIR / "input"
    OUTPUT_DIR: Path = DATA_DIR / "output"
    INTERMEDIATE_DIR: Path = DATA_DIR / "intermediate"
    
    INPUT_SAMPLES: Path = INPUT_DIR / "samples"
    INPUT_BLANKS: Path = INPUT_DIR / "blanks"
    INPUT_CALIBRANTS: Path = INPUT_DIR / "calibration"
    INPUT_DATABASES: Path = INPUT_DIR / "databases"
    
    INTERMEDIATE_SAMPLES: Path = INTERMEDIATE_DIR / "samples"
    INTERMEDIATE_CALIBRANTS: Path = INTERMEDIATE_DIR / "calibrants"

@dataclass
class PeakDetectionConfig:
    """Configuration pour la détection des pics"""
    threshold: int = 100
    smooth_iterations: int = 7
    smooth_radius: Dict[str, int] = field(default_factory=lambda: {
        'mz': 0,
        'drift_time': 1,
        'retention_time': 0
    })
    peak_radius: Dict[str, int] = field(default_factory=lambda: {
        'mz': 2,
        'drift_time': 10,
        'retention_time': 0
    })

@dataclass
class IntraClusteringConfig:
    """Configuration pour le clustering au sein d'un même échantillon"""
    mz_ppm: float = 10
    dt_tolerance: float = 1.0
    rt_tolerance: float = 0.1
    dbscan_eps: float = 1.0
    dbscan_min_samples: int = 1
    algorithm: str = 'kd_tree'

@dataclass
class FeatureAlignmentConfig:
    """Configuration pour l'alignement des features entre échantillons"""
    mz_ppm: float = 10
    dt_tolerance: float = 1.02
    rt_tolerance: float = 0.2
    dbscan_eps: float = 1.0
    dbscan_min_samples: int = 1
    algorithm: str = 'kd_tree'

@dataclass
class BlankSubtractionConfig:
    """Configuration pour la soustraction du blank"""
    mz_ppm: float = 10
    dt_tolerance: float = 0.22
    rt_tolerance: float = 0.1
    dbscan_eps: float = 1.5
    dbscan_min_samples: int = 2
    cluster_ratio: float = 0.5
    
    
@dataclass
class BlankReplicateConfig:
    """Configuration pour le clustering des réplicats de blanks"""
    mz_ppm: float = 10
    dt_tolerance: float = 0.22  
    rt_tolerance: float = 0.1   
    dbscan_eps: float = 0.6
    dbscan_min_samples: int = 2
    algorithm: str = 'kd_tree'

@dataclass
class ReplicateConfig:
    """Configuration pour le traitement des réplicats"""
    min_replicates: int = 2  
    mz_ppm: float = 10
    dt_tolerance: float = 0.22
    rt_tolerance: float = 0.1
    dbscan_eps: float = 0.6
    dbscan_min_samples: int = 2
    algorithm: str = 'kd_tree'

@dataclass
class IdentificationConfig:
    """Configuration pour l'identification des composés"""
    database_file: str = "norman_all_ccs_all_rt_pos_neg_with_ms2.h5"
    database_key: str = "positive"
    tolerances: Dict[str, float] = field(default_factory=lambda: {
        'mz_ppm': 5,
        'ccs_percent': 8,
        'rt_min': 0.3
    })
    weights: Dict[str, float] = field(default_factory=lambda: {
        'mz': 0.4,
        'ccs': 0.4,
        'rt': 0.2
    })
    ms2_score_threshold: float = 0.2

@dataclass
class MS2ExtractionConfig:
    """Configuration pour l'extraction MS2"""
    rt_tolerance: float = 0.00422  # minutes
    dt_tolerance: float = 0.22     # ms
    mz_round_decimals: int = 3
    max_peaks: int = 10
    intensity_scale: int = 999

@dataclass
class ProcessingConfig:
    """Configuration générale du traitement"""
    parallel_max_workers: int = 2
    batch_size: int = 2
    intensity_threshold: float = 100.0

class Config:
    """Configuration centralisée pour l'ensemble du pipeline."""
    
    # Chemins
    PATHS = PathConfig()
    
    # Configurations spécifiques
    PEAK_DETECTION = PeakDetectionConfig()
    INTRA_CLUSTERING = IntraClusteringConfig()
    BLANK_REPLICATE = BlankReplicateConfig()
    FEATURE_ALIGNMENT = FeatureAlignmentConfig()
    BLANK_SUBTRACTION = BlankSubtractionConfig()
    REPLICATE = ReplicateConfig()
    IDENTIFICATION = IdentificationConfig()
    MS2_EXTRACTION = MS2ExtractionConfig()
    PROCESSING = ProcessingConfig()
    
    # Mapping des colonnes de la base de données
    DB_COLUMNS: Dict[str, str] = {
        'name': 'Name',
        'mz': 'mz',
        'ccs_exp': 'ccs_exp',
        'ccs_pred': 'ccs_pred',
        'rt_obs': 'Observed_RT',
        'rt_pred': 'Predicted_RT'
    }

    @classmethod
    def setup_directories(cls) -> None:
        """Crée tous les répertoires nécessaires."""
        dirs = [
            cls.PATHS.INPUT_DIR,
            cls.PATHS.INPUT_SAMPLES,
            cls.PATHS.INPUT_BLANKS,
            cls.PATHS.INPUT_CALIBRANTS,
            cls.PATHS.INPUT_DATABASES,
            cls.PATHS.INTERMEDIATE_DIR,
            cls.PATHS.INTERMEDIATE_SAMPLES,
            cls.PATHS.INTERMEDIATE_CALIBRANTS,
            cls.PATHS.OUTPUT_DIR
        ]
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_sample_path(cls, sample_name: str) -> Path:
        """Retourne le chemin complet pour un échantillon."""
        return cls.PATHS.INPUT_SAMPLES / f"{sample_name}.parquet"
    
    @classmethod
    def get_intermediate_path(cls, sample_name: str, subfolder: str = "") -> Path:
        """Retourne le chemin intermédiaire pour un échantillon."""
        path = cls.PATHS.INTERMEDIATE_SAMPLES / sample_name
        if subfolder:
            path = path / subfolder
        return path
    
    @classmethod
    def get_output_path(cls, filename: str) -> Path:
        """Retourne le chemin de sortie pour un fichier."""
        return cls.PATHS.OUTPUT_DIR / filename
