# 🔬 Pipeline d'Identification MS

Pipeline d'identification de composés intégrant MS1, mobilité ionique (CCS) et MS2.

## ⚡ Détection des pics

Extraction des pics selon trois dimensions analytiques :
- Masse exacte (m/z)
- Temps de rétention chromatographique (RT)
- Mobilité ionique (DT)

## 🎯 Tolérances d'identification

L'identification intègre quatre niveaux d'information analytique :
- Masse exacte : ± 5 ppm
- Section efficace de collision : ± 8%
- Temps de rétention : ± 2 min
- Comparaison des spectres MS2

## 📊 Résultats types

| Composé | m/z mesuré | m/z théorique | Erreur (ppm) | CCS | RT (min) | Score MS2 | Formule | Adduit |
|---------|------------|---------------|--------------|-----|----------|-----------|----------|---------|
| Caféine | 195.0879 | 195.0882 | -1.11 | 143.97 | 4.01 | 0.89 | C8H10N4O2 | [M+H]+ |
| Paracétamol | 152.0706 | 152.0712 | -3.94 | 131.45 | 3.22 | 0.92 | C8H9NO2 | [M+H]+ |
| Ibuprofène | 207.1378 | 207.1380 | -0.96 | 152.88 | 8.45 | 0.78 | C13H18O2 | [M+H]+ |

## 📦 Installation

1. Cloner le repository :
```bash
git clone https://github.com/narvall018/pipeline_identification_ms.git
cd pipeline_identification_ms
```

2. Créer l'environnement conda :
```bash
conda env create -f environment.yml -v # 5min
conda activate ms_pipeline
```

3. Vérifier l'installation :
```bash
python -c "import deimos; print(deimos.__version__)"
```

4. ⚠️ Base de données de référence (requis) :
   - Base de données NORMAN [📥 Télécharger ici](https://drive.google.com/file/d/1mZa1r9RZ4Ioy1cILJqIteAz3vUs_UIaU/view?usp=drive_link)
   - Créer un dossier `databases` dans `data/input/`
   - Copier le fichier `norman_all_ccs_all_rt_pos_neg_with_ms2.h5` dans `data/input/databases/`

## 📁 Structure du Projet

```
pipeline_identification_ms/
├── data/
│   ├── input/
│   │   ├── samples/          # Fichiers .parquet
│   │   ├── calibration/      # Calibration CCS
│   │   └── databases/        # Base de données
│   ├── intermediate/         # Données intermédiaires
│   └── output/              # Résultats
├── logs/
├── scripts/
│   ├── config/             # Configuration
│   ├── processing/         # Traitement
│   ├── utils/             # Utilitaires
│   └── visualization/     # Visualisation
└── tests/
```

## 🚀 Utilisation

1. Placer les fichiers :
   - `.parquet` dans `data/input/samples/`
   - Calibration dans `data/input/calibration/`
   - Base de données dans `data/input/databases/`

2. Exécuter :
```bash
python main.py
```

## ⚙️ Configuration

Configuration dans `scripts/config/config.py` :
```python
IDENTIFICATION = {
    'tolerances': {
        'mz_ppm': 5,
        'ccs_percent': 8,
        'rt_min': 2
    }
}
```

## 🐛 Dépannage

En cas d'erreur DEIMoS :
```bash
conda activate ms_pipeline
pip uninstall deimos
pip install git+https://github.com/pnnl/deimos.git
```

## 📝 Logging

Fichiers logs : `logs/peak_detection.log` 

