# 🔬 Pipeline d'Identification MS


## Table des matières
1. [Introduction](#1-introduction-)
2. [Installation](#2-installation-%EF%B8%8F)
3. [Structure et Configuration](#3-structure-et-configuration-)
4. [Composants Principaux](#4-composants-principaux-)
5. [Utilisation](#5-utilisation-)
6. [Résultats et Visualisation](#6-résultats-et-visualisation-)
7. [Licence](#7-licence-%E2%9A%96%EF%B8%8F)
8. [Citation](#8-citation)


## 1. Introduction 🚀

Pipeline d'identification de composés intégrant MS1, mobilité ionique (CCS) et MS2, avec filtration par réplicats, soustraction des blancs, et alignement des échantillons.

La pipepline travaille sur trois dimensions analytiques :
- La masse exacte (m/z)
- Le temps de rétention (RT)  
- Le temps de dérive (DT)

La pipeline est structurée en modules interconnectés, chacun responsable d'une étape spécifique du traitement des données.

**Flux de Données**

1. Détection des pics MS1 dans les données brutes
2. Traitement des réplicats pour valider les pics détectés
3. Soustraction des blancs pour éliminer les contaminations
4. Calcul des valeurs CCS via la calibration
5. Alignement des features entre échantillons
6. Identification des composés et validation MS2

**Composants Principaux**

- **PeakDetector** : détection des pics dans les données brutes
- **ReplicateProcessor** : gestion et validation des réplicats
- **BlankProcessor** : traitement et soustraction des blancs
- **CCSCalibrator** : calibration et calcul des CCS
- **FeatureProcessor** : alignement des features entre échantillons
- **CompoundIdentifier** : identification des composés
- **MS2Extractor** : extraction et validation des spectres MS2

Chaque composant peut être utilisé indépendamment ou dans le flux complet de la pipeline.


## 2. Installation ⚙️

### 2.1. Via Conda (Recommandé)
```bash
# Cloner le repository
git clone https://github.com/votre_username/pipeline_identification_ms.git
cd pipeline_identification_ms

# Créer et activer l'environnement
conda env create -f environment.yml
conda activate ms_pipeline

# Vérifier l'installation
python -c "import deimos; print(deimos.__version__)"
```

En cas d'erreur avec DEIMoS :
```bash
conda activate ms_pipeline
pip uninstall deimos
pip install git+https://github.com/pnnl/deimos.git
```

### 2.2. Via Pip
```bash
# Créer un environnement virtuel
python -m venv ms_env

# Activer l'environnement
# Sur Windows :
ms_env\Scripts\activate
# Sur Linux/macOS :
source ms_env/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

### 2.3. Prérequis

**Fichiers Requis**
- 📥 Base de données NORMAN ([Télécharger ici](https://drive.google.com/file/d/1mZa1r9RZ4Ioy1cILJqIteAz3vUs_UIaU/view))
- 🗂️ Fichiers blancs dans `data/input/blanks/`

**Structure à Créer**
```
data/
├── input/
│   ├── samples/          # Vos fichiers .parquet
│   ├── blanks/          # Vos fichiers blancs
│   ├── calibration/     # Fichiers de calibration
│   └── databases/       # Base NORMAN
```

**Configuration Système**
- Python 3.8 ou supérieur
- 8 Go RAM minimum recommandé
  

## 3. Structure et Configuration 📁

### 3.1. Organisation du Projet

```
pipeline_identification_ms/
├── data/
│   ├── input/  
│   │   ├── samples/        # Fichiers d'échantillons (.parquet)
│   │   ├── blanks/         # Fichiers de blancs
│   │   ├── calibration/    # Fichiers de calibration CCS
│   │   └── databases/      # Base de données NORMAN
│   │
│   ├── intermediate/       # Résultats intermédiaires
│   │   └── samples/
│   │       └── nom_echantillon/
│   │           ├── ms1/
│   │           │   ├── peaks_before_blank.parquet
│   │           │   └── common_peaks.parquet
│   │           └── ms2/
│   │               └── spectra.parquet
│   │
│   └── output/            # Résultats finaux
│       ├── feature_matrix.parquet
│       ├── feature_matrix.csv
│       └── features_complete.parquet
│
├── scripts/               # Code source
│   ├── processing/        # Modules de traitement
│   │   ├── peak_detection.py
│   │   ├── blank_processing.py
│   │   ├── replicate_processing.py
│   │   ├── ccs_calibration.py
│   │   ├── feature_matrix.py
│   │   ├── identification.py
│   │   └── ms2_extraction.py
│   │
│   └── utils/            # Fonctions utilitaires
│       ├── io_handlers.py
│       ├── matching_utils.py
│       └── replicate_handling.py
│
└── logs/                 # Fichiers de logs
```

**Description des Dossiers**

- `data/input/` : Contient toutes les données d'entrée nécessaires
  - `samples/` : Fichiers d'échantillons au format .parquet
  - `blanks/` : Fichiers de blancs analytiques
  - `calibration/` : Données pour la calibration CCS
  - `databases/` : Base de données de référence

- `data/intermediate/` : Stocke les résultats de chaque étape
  - Organisation par échantillon
  - Séparation MS1/MS2
  - Résultats avant/après soustraction des blancs

- `data/output/` : Contient les résultats finaux
  - Matrices de features
  - Identifications
  - Fichiers aux formats .parquet et .csv

- `scripts/` : Code source de la pipeline
  - `processing/` : Modules principaux de traitement
  - `utils/` : Fonctions utilitaires et helpers


- Fichiers d'échantillons : `nom_echantillon.parquet`
- Réplicats : `nom_echantillon_replicate_1_X.parquet`
- Blancs : `blank_replicate_1.parquet`
- Résultats : `nom_explicite.parquet/csv`


### 3.2. Configuration

La configuration de la pipeline est gérée par des classes dédiées dans `config.py`. Chaque aspect du traitement a sa propre configuration avec des paramètres par défaut modifiables.

**Paramètres Globaux**
- Organisation en classes de configuration
- Gestion des chemins automatisée
- Configuration du traitement parallèle
- Paramètres clustering

**Tolérances d'Identification**
- Paramètres MS1 et MS2
- Tolérances pour l'alignement
- Seuils de validation

**Optimisation des Performances**
- Nombre de workers parallèles
- Taille des lots de traitement
- Seuils d'intensité

```python
class Config:
    """Configuration du pipeline"""
    
    # Détection des pics
    PEAK_DETECTION = PeakDetectionConfig(
        threshold = 100,
        smooth_iterations = 7,
        smooth_radius = {
            'mz': 0,
            'drift_time': 1,
            'retention_time': 0
        },
        peak_radius = {
            'mz': 2,
            'drift_time': 10,
            'retention_time': 0
        }
    )

    # Identification des composés
    IDENTIFICATION = IdentificationConfig(
        database_file = "norman_all_ccs_all_rt_pos_neg_with_ms2.h5",
        database_key = "positive",
        tolerances = {
            'mz_ppm': 5,
            'ccs_percent': 8,
            'rt_min': 0.1
        }
    )

    # Soustraction des blancs
    BLANK_SUBTRACTION = BlankSubtractionConfig(
        mz_ppm = 10,
        dt_tolerance = 0.22,
        rt_tolerance = 0.1,
        dbscan_eps = 1.5,
        dbscan_min_samples = 2
    )

    # Traitement des réplicats
    REPLICATE = ReplicateConfig(
        min_replicates = 2,
        mz_ppm = 10,
        dt_tolerance = 0.22,
        rt_tolerance = 0.1
    )

    # Extraction MS2
    MS2_EXTRACTION = MS2ExtractionConfig(
        rt_tolerance = 0.00422,
        dt_tolerance = 0.22,
        max_peaks = 10,
        intensity_scale = 999
    )
```

**Modules Configurables**
- PeakDetection : détection des pics MS1
- Identification : paramètres d'identification
- BlankSubtraction : soustraction des blancs
- Replicate : gestion des réplicats
- MS2Extraction : extraction des spectres MS2
- Alignement : Alignement des échantillons


## 4. Composants Principaux 🔧

### 4.1. Détection des Pics (PeakDetector)

Le PeakDetector est responsable de la détection des pics MS1 dans les données brutes. Il intègre le lissage des données, la détection des pics et le clustering intra-échantillon.

**Utilisation Basique**
```python
from scripts.processing.peak_detection import PeakDetector

# Initialisation
detector = PeakDetector()

# Traitement simple
peaks = detector.process_sample(data)

# Traitement avec paramètres personnalisés
peaks = detector.detect_peaks(
    data,
    threshold=200,              # Seuil plus élevé
    smooth_iterations=7         # Plus d'itérations de lissage
)
```

**Paramètres**
```python
# Configuration par défaut
PEAK_DETECTION = {
    'threshold': 50,           # Seuil d'intensité minimum
    'smooth_iterations': 7,     # Nombre d'itérations pour le lissage
    'smooth_radius': {          # Rayons de lissage
        'mz': 0,
        'drift_time': 1,
        'retention_time': 0
    },
    'peak_radius': {           # Rayons pour la détection
        'mz': 2,
        'drift_time': 10,
        'retention_time': 0
    }
}
```

**Workflow de Traitement**
1. Préparation des données
   ```python
   prepared_data = detector.prepare_data(data)
   ```

2. Détection des pics
   ```python
   peaks = detector.detect_peaks(prepared_data)
   ```

3. Clustering des pics
   ```python
   clustered_peaks = detector.cluster_peaks(peaks)
   ```

**Format des Données de Sortie**
```python
# Exemple de DataFrame retourné
peaks_df = {
    'mz': [123.4567, ...],           # Masse exacte
    'drift_time': [12.3, ...],       # Temps de dérive
    'retention_time': [5.6, ...],    # Temps de rétention
    'intensity': [1000, ...]         # Intensité du pic
}
```

### 4.2. Traitement des Réplicats (ReplicateProcessor)

Le ReplicateProcessor valide la présence des pics entre différents réplicats d'un même échantillon pour assurer la fiabilité des résultats.

**Utilisation Basique**
```python
from scripts.processing.replicate_processing import ReplicateProcessor
from pathlib import Path

# Initialisation
processor = ReplicateProcessor()

# Liste des fichiers réplicats
replicate_files = [
    Path("data/input/samples/sample_replicate_1.parquet"),
    Path("data/input/samples/sample_replicate_1_2.parquet"),
    Path("data/input/samples/sample_replicate_1_3.parquet")
]

# Traitement des réplicats
results = processor.process_sample_with_replicates(
    sample_name="mon_echantillon",
    replicate_files=replicate_files,
    output_dir=Path("data/output")
)
```

**Configuration**
```python
REPLICATE = {
    'min_replicates': 2,       # Nombre minimum de réplicats requis
    'mz_ppm': 10,             # Tolérance m/z en ppm
    'dt_tolerance': 0.22,      # Tolérance temps de dérive
    'rt_tolerance': 0.1,       # Tolérance temps de rétention
    'dbscan_eps': 0.6,        # Epsilon pour DBSCAN
    'dbscan_min_samples': 2    # Échantillons minimum pour DBSCAN
}
```

**Workflow de Traitement**
1. Traitement individuel des réplicats
```python
peaks_dict, initial_counts = processor.process_replicates(replicate_files)
```

2. Clustering entre réplicats
```python
common_peaks = processor.cluster_replicates(peaks_dict)
```

**Format de Sortie**
```python
# Structure des résultats
results = {
    'mz': [],                 # Masses moyennes
    'drift_time': [],         # Temps de dérive moyens
    'retention_time': [],     # Temps de rétention moyens
    'intensity': [],          # Intensités maximales
    'n_replicates': []        # Nombre de réplicats où le pic est trouvé
}
```

### 4.3. Soustraction des Blancs (BlankProcessor)

Le BlankProcessor élimine les contaminations et le bruit de fond en soustrayant les pics détectés dans les blancs analytiques.

**Utilisation Basique**
```python
from scripts.processing.blank_processing import BlankProcessor
from pathlib import Path

# Initialisation
blank_processor = BlankProcessor()

# Traitement d'un fichier blank individuel
blank_peaks = blank_processor.process_blank_file(blank_file)

# Soustraction des blancs des échantillons
clean_peaks = blank_processor.subtract_blank_peaks(sample_peaks, blank_peaks)
```

**Méthodologie**
1. Traitement des Blancs
   - Détection des pics dans les blancs
   - Regroupement des réplicats de blancs
   - Validation des pics communs

2. Soustraction
   - Comparaison des pics échantillon/blanc
   - Application des tolérances
   - Filtrage des pics contaminants

**Configuration**
```python
BLANK_SUBTRACTION = {
    'mz_ppm': 10,              # Tolérance masse
    'dt_tolerance': 0.22,      # Tolérance temps de dérive
    'rt_tolerance': 0.1,       # Tolérance temps de rétention
    'dbscan_eps': 1.5,         # Epsilon clustering
    'dbscan_min_samples': 2,   # Minimum échantillons
}
```

**Format des Données de Sortie**
```python
clean_peaks = {
    'mz': [],                  # Masses validées
    'drift_time': [],          # Temps de dérive
    'retention_time': [],      # Temps de rétention
    'intensity': []            # Intensités
}
```

### 4.4. Features et Alignement (FeatureProcessor)

Le FeatureProcessor aligne les pics entre différents échantillons pour créer une matrice de features commune. Il permet d'identifier et de quantifier les composés à travers tous les échantillons.

**Utilisation Basique**
```python
from scripts.processing.feature_matrix import FeatureProcessor
from pathlib import Path

# Initialisation
processor = FeatureProcessor()

# Alignement et création de matrice
matrix, feature_info, raw_files = processor.align_features_across_samples(
    samples_dir=Path("data/intermediate/samples")
)

# Traitement des features avec identification
identifications = processor.process_features(
    feature_df=feature_info,
    raw_files=raw_files,
    identifier=identifier  # Instance de CompoundIdentifier
)
```

**Configuration**
```python
FEATURE_ALIGNMENT = {
    'mz_ppm': 10,              # Tolérance masse
    'dt_tolerance': 1.02,      # Tolérance temps de dérive
    'rt_tolerance': 0.2,       # Tolérance temps de rétention
    'dbscan_eps': 1.0,         # Epsilon pour clustering
    'dbscan_min_samples': 1    # Minimum d'échantillons
}
```

**Création de Matrices**
1. Matrice d'Intensités
```python
# Format de la matrice
matrix_df = pd.DataFrame(
    index=sample_names,        # Échantillons en lignes
    columns=feature_names      # Features en colonnes
)
```

2. Matrice de Features
```python
# Informations sur les features
feature_df = {
    'mz': [],                  # Masse moyenne
    'retention_time': [],      # RT moyen
    'drift_time': [],          # DT moyen
    'intensity': [],           # Intensité maximale
    'n_samples': [],           # Nombre d'échantillons
    'feature_id': []           # Identifiant unique
}
```

**Sorties Générées**
```
data/output/
├── feature_matrix.parquet     # Matrice d'intensités
├── feature_matrix.csv        
├── features_complete.parquet  # Informations détaillées
└── features_complete.csv
```


### 4.5. Calibration CCS (CCSCalibrator)

Le CCSCalibrator permet de calculer les valeurs CCS (Section Efficace de Collision) à partir des temps de dérive mesurés.

**Utilisation Basique**
```python
from scripts.processing.ccs_calibration import CCSCalibrator

# Initialisation avec fichier de calibration
calibrator = CCSCalibrator("path/to/calibration.csv")

# Calibration
calibrator.calibrate()

# Calcul des CCS pour les pics
ccs_values = calibrator.calculate_ccs(peaks_df)
```

**Processus de Calibration**
1. Chargement des données
```python
# Format du fichier de calibration
calibration_data = {
    'Reference m/z': [],    # m/z de référence
    'Measured m/z': [],     # m/z mesuré
    'Measured Time': [],    # Temps de dérive mesuré
    'Reference rCCS': [],   # CCS de référence
    'z': []                # État de charge
}
```

### 4.6. Identification (CompoundIdentifier)

Le CompoundIdentifier compare les pics détectés avec une base de données de référence pour identifier les composés.

**Utilisation Basique**
```python
from scripts.processing.identification import CompoundIdentifier

# Initialisation
identifier = CompoundIdentifier()

# Identification des composés
matches = identifier.identify_compounds(peaks_df, output_dir)
```

**Configuration Base de Données**
```python
IDENTIFICATION = {
    'database_file': "norman_all_ccs_all_rt_pos_neg_with_ms2.h5",
    'database_key': "positive",
    'tolerances': {
        'mz_ppm': 5,           # Tolérance masse
        'ccs_percent': 8,      # Tolérance CCS
        'rt_min': 0.1          # Tolérance RT
    },
    'ms2_score_threshold': 0.5 # Seuil score MS2
}
```

**Format des Résultats**
```python
matches = {
    'match_name': [],         # Nom du composé
    'formula': [],           # Formule moléculaire
    'confidence_level': [],   # Niveau de confiance
    'global_score': [],      # Score global
    'ms2_score': []          # Score MS2 si disponible
}
```

### 4.7. MS2 (MS2Extractor)

Le MS2Extractor extrait et traite les spectres MS2 pour valider les identifications via la comparaison avec des spectres de référence.

**Utilisation Basique**
```python
from scripts.processing.ms2_extraction import MS2Extractor

# Initialisation
extractor = MS2Extractor()

# Extraction d'un spectre MS2
spectra = extractor.extract_ms2_spectrum(
    ms2_data=ms2_data,
    rt=retention_time,
    dt=drift_time
)

# Extraction pour plusieurs matches
matches_with_ms2 = extractor.extract_ms2_for_matches(
    matches_df=matches,
    raw_parquet_path="path/to/raw.parquet",
    output_dir="path/to/output"
)
```

**Configuration**
```python
MS2_EXTRACTION = {
    'rt_tolerance': 0.00422,    # Fenêtre RT (minutes)
    'dt_tolerance': 0.22,       # Fenêtre DT (ms)
    'mz_round_decimals': 3,     # Précision m/z
    'max_peaks': 10,            # Pics maximum
    'intensity_scale': 999      # Échelle d'intensité
}
```

## 5. Utilisation 📊

Pour utiliser la pipeline, il suffit de suivre ces étapes :

**1. Préparation des Données**

Placez vos fichiers dans les dossiers appropriés :
```
data/
├── input/
│   ├── samples/          # Vos échantillons (.parquet)
│   ├── blanks/          # Vos blancs
│   ├── calibration/     # Fichier de calibration CCS
│   └── databases/       # Base NORMAN
```

**2. Lancement de la Pipeline**

Depuis le terminal, dans le dossier du projet :
```bash
# Activation de l'environnement
conda activate ms_pipeline

# Lancement de la pipeline
python main.py
```

**3. Options Disponibles**
```bash
# Aide sur les options
python main.py --help

# Spécifier un dossier d'entrée différent
python main.py --input_dir "chemin/vers/données"

# Spécifier un dossier de sortie
python main.py --output_dir "chemin/vers/sortie"
```

**4. Suivi du Traitement**

La pipeline affiche sa progression :
```
🚀 Démarrage du traitement...
   
📊 Traitement des blancs...
✓ Blanc 1 traité
✓ Blanc 2 traité

🔍 Traitement des échantillons...
✓ Échantillon 1 (3 réplicats)
✓ Échantillon 2 (3 réplicats)
...

✨ Traitement terminé !
```

**5. Résultats**

Les résultats sont automatiquement sauvegardés dans `data/output/` :
- `feature_matrix.parquet` : Matrice d'intensités
- `feature_matrix.csv` : Version CSV de la matrice
- `features_complete.parquet` : Données complètes avec identifications
- `features_complete.csv` : Version CSV des identifications


## 6. Résultats et Visualisation 📈

### 6.1. Formats de Sortie

La pipeline génère plusieurs fichiers de sortie organisés de manière structurée.

**Structure des Dossiers de Sortie**
```
data/output/
├── feature_matrix.parquet     # Matrice principale des features
├── feature_matrix.csv        
├── features_complete.parquet  # Données avec identifications
└── features_complete.csv     
```

**1. Matrice des Features**
- Format : CSV et Parquet
- Structure :
```python
# feature_matrix.csv
              F001_mz123.45  F002_mz456.78  F003_mz789.01
Sample_1         1234.56       0.00           789.01
Sample_2         1567.89       456.78         0.00
Sample_3         1432.12       567.89         654.32
```

**2. Liste des Identifications**
- Format : CSV et Parquet
- Contenu :
```python
# features_complete.csv
Colonnes principales :
- feature_id       # Identifiant unique
- mz              # Masse mesurée
- rt              # Temps de rétention
- drift_time      # Temps de dérive
- ccs             # Valeur CCS calculée
- match_name      # Nom du composé identifié
- formula         # Formule moléculaire
- score           # Score global
- confidence      # Niveau de confiance
```

**Exemple de Résultats**
| Composé | m/z mesuré | m/z théorique | Erreur (ppm) | CCS | RT (min) | Score MS2 | Formule | Adduit |
|---------|------------|---------------|--------------|-----|-----------|-----------|----------|---------|
| Caféine | 195.0879 | 195.0882 | -1.11 | 143.97 | 4.01 | 0.89 | C8H10N4O2 | [M+H]+ |
| Paracétamol | 152.0706 | 152.0712 | -3.94 | 131.45 | 3.22 | 0.92 | C8H9NO2 | [M+H]+ |
| Ibuprofène | 207.1378 | 207.1380 | -0.96 | 152.88 | 8.45 | 0.78 | C13H18O2 | [M+H]+ |

**Formats Disponibles**
- `.parquet` : Format optimisé pour l'analyse ultérieure
- `.csv` : Format lisible et compatible avec Excel


**Accès aux Résultats**
```python
# Lecture des résultats
import pandas as pd

# Matrice des features
matrix = pd.read_parquet("data/output/feature_matrix.parquet")

# Identifications complètes
identifications = pd.read_parquet("data/output/features_complete.parquet")
```

**Analyse des Résultats**

La matrice de features (`feature_matrix.csv/parquet`) contient :
- Les intensités de chaque feature dans tous les échantillons
- Format : échantillons en lignes, features en colonnes
- Nomenclature : `FXXX_mzYYY.YYYY` où :
  - XXX : numéro unique de la feature
  - YYY.YYYY : masse exacte mesurée

Dans `features_complete.csv/parquet` :
- Toutes les features identifiées
- Paramètres analytiques (m/z, RT, DT, CCS)
- Informations d'identification (nom, formule, score)


## 7. Licence ⚖️

Ce projet est sous licence [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

### 8. Citation

Pour citer ce projet dans une publication académique, veuillez utiliser :

```bibtex
@software{pipeline_identification_ms,
    title = {Pipeline d'Identification MS},
    year = {2024},
    author = {Sade, Julien},
    url = {https://github.com/narvall018/pipeline_identification_ms},
    version = {1.0.0},
    institution = {Leesu},
    note = {Pipeline pour l'identification de composés par spectrométrie de masse}
}
```

Pour une citation dans le texte :
> Sade, J. (2024). Pipeline d'Identification MS,https://github.com/narvall018/pipeline_identification_ms

**Contact**

Pour toute question concernant l'utilisation :
- ✉️ julien.sade@u-pec.fr

