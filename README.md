# Pipeline d'Identification MS

Pipeline d'analyse pour l'identification de composés à partir de données de spectrométrie de masse. Cette pipeline intègre:
- Détection de pics MS1 (utilisant DEIMoS)
- Calibration CCS
- Identification de composés
- Analyse MS2 et calcul de scores de similarité

## Prérequis

- Git
- Python 3.9+
- Conda ou Miniconda

## Installation Rapide

```bash
# Cloner le repository
git clone [URL_DU_REPO]
cd pipeline_identification

# Créer et activer l'environnement (installe automatiquement toutes les dépendances y compris DEIMoS)
conda env create -f environment.yml
conda activate ms_pipeline

# Vérifier l'installation
python -c "import deimos; print(deimos.__version__)"
```

## Structure du Projet

```
pipeline_identification/
├── data/
│   ├── input/
│   │   ├── samples/          # Fichiers d'échantillons (.parquet)
│   │   ├── calibration/      # Données de calibration CCS
│   │   └── database/         # Base de données de référence
│   ├── intermediate/         # Données intermédiaires générées
│   └── output/              # Résultats finaux et visualisations
├── logs/                    # Fichiers de logs
├── scripts/
│   ├── config/             # Configuration
│   ├── processing/         # Scripts de traitement
│   ├── utils/             # Fonctions utilitaires
│   └── visualization/     # Scripts de visualisation
└── tests/                 # Tests unitaires
```

## Utilisation

1. Placer les fichiers d'entrée:
   - Fichiers .parquet dans `data/input/samples/`
   - Données de calibration dans `data/input/calibration/`
   - Base de données dans `data/input/database/`

2. Lancer la pipeline:
```bash
python main.py
```

## Pipeline de Traitement

1. **Détection des pics (DEIMoS)**
   - Lecture des données brutes (.parquet)
   - Préparation des données MS1
   - Détection et clustering des pics

2. **Calibration CCS**
   - Calcul des valeurs CCS
   - Application de la calibration

3. **Identification des composés**
   - Recherche dans la base de données
   - Calcul des scores de correspondance
   - Attribution des niveaux de confiance

4. **Analyse MS2**
   - Extraction des spectres MS2
   - Comparaison avec la base de données
   - Calcul des scores de similarité

## Configuration

Les paramètres de la pipeline sont configurables dans `scripts/config/config.py`:
- Tolérances pour l'identification
- Paramètres de détection des pics
- Critères de confiance
- Chemins des fichiers

## Développement

### Tests
```bash
python -m pytest tests/
```

### Logging
Les logs sont enregistrés dans `logs/peak_detection.log`

## Résolution des problèmes courants

Si vous rencontrez des problèmes avec DEIMoS :
```bash
conda activate ms_pipeline
pip uninstall deimos
pip install git+https://github.com/pnnl/deimos.git
```

## Dépendances principales

- DEIMoS: Pour la détection des pics et le traitement MS
- NumPy: Pour les calculs numériques
- Pandas: Pour la manipulation des données
- Scikit-learn: Pour le clustering et le machine learning
- SciPy: Pour les calculs scientifiques
