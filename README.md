# 🔬 Pipeline d'Identification MS

Pipeline d'identification de composés intégrant MS1, mobilité ionique (CCS) et MS2, avec filtration par réplicats, soustraction des blancs, et alignement des échantillons.

## ⚡ Détection des pics et Prétraitement

Extraction des pics selon trois dimensions analytiques :

- 📐 **Masse exacte (m/z)**
- ⏱️ **Temps de rétention chromatographique (RT)**
- 🌐 **Mobilité ionique (DT)**

### 🔄 Pipeline de Traitement

Le pipeline de traitement comprend les étapes suivantes :

1. Détection des pics dans les données des échantillons
2. Filtration des réplicats pour ne conserver que les pics présents dans un nombre minimum de réplicats
3. Soustraction des blancs pour éliminer les pics présents dans les échantillons blancs
4. Groupement des pics via l'algorithme DBSCAN pour aligner les pics correspondants entre les échantillons
5. Computation des CCS
6. Identification des composés en comparant les pics groupés à une base de données de référence, en utilisant des tolérances définies pour la masse exacte, le temps de rétention et la CCS
7. Extraction des spectres MS2 pour les composés identifiés
8. Comparaison des spectres MS2 entre les échantillons pour évaluer la confiance dans l'identification

Chaque échantillon subit une filtration par réplicats et une soustraction des blancs pour améliorer la fiabilité des données.

## 📦 Prérequis
### 🗂️ Fichiers Blancs (requis)
Les fichiers blancs doivent être placés dans `data/input/blanks/`.

### ⚠️ Base de Données de Référence (requis)

- Base de données NORMAN 📥 [Télécharger ici](https://drive.google.com/file/d/1mZa1r9RZ4Ioy1cILJqIteAz3vUs_UIaU/view)
- Créer un dossier `databases` dans `data/input/`
- Copier le fichier `norman_all_ccs_all_rt_pos_neg_with_ms2.h5` dans `data/input/databases/`

### 🎯 Tolérances d'Identification
L'identification intègre quatre niveaux d'information analytique :

🧪 Masse exacte : ± 5 ppm
🌐 Section efficace de collision (CCS) : ± 8%
⏱️ Temps de rétention : ± 2 min
📊 Comparaison des spectres MS2

## 📊 Matrices et Features
La pipeline génère plusieurs fichiers de sortie :

🗃️ Matrice de Features avec Intensités (`feature_matrix.csv/parquet`)
- Features triées par taux de remplissage décroissant.
- Nomenclature :
  - Si identifié : `[nom_composé]_mz[XXX]_rt[XX]_dt[XX]_ccs[XXX]`
  - Sinon : `mz[XXX]_rt[XX]_dt[XX]_ccs[XXX]`

📈 Statistiques de Clustering (`cluster_means.csv`)
- Moyennes des paramètres par cluster.
- Identification associée quand disponible.

📊 Statistiques de la Matrice (`feature_matrix_stats.csv`)
- Nombre total de features.
- Taux de remplissage.
- Statistiques par échantillon/feature.

## 📊 Visualisations Générées
Des visualisations sont générées dans `data/output/`, incluant :

- Chromatogrammes TIC
- Similarités  
- Nombre de molécules détectées

## 📊 Résultats Types
Composé | m/z mesuré | m/z théorique | Erreur (ppm) | CCS | RT (min) | Score MS2 | Formule | Adduit
--- | --- | --- | --- | --- | --- | --- | --- | ---
Caféine | 195.0879 | 195.0882 | -1.11 | 143.97 | 4.01 | 0.89 | C8H10N4O2 | [M+H]+
Paracétamol | 152.0706 | 152.0712 | -3.94 | 131.45 | 3.22 | 0.92 | C8H9NO2 | [M+H]+
Ibuprofène | 207.1378 | 207.1380 | -0.96 | 152.88 | 8.45 | 0.78 | C13H18O2 | [M+H]+

## 📦 Installation
### 🐉🍃 Environnement Conda (recommandé)
1. Cloner le repository 
   ```bash
   git clone https://github.com/narvall018/pipeline_identification_ms.git
   cd pipeline_identification_ms  
   ```
2. Créer l'environnement conda
   ```bash
   conda env create -f environment.yml -v
   conda activate ms_pipeline
   ```
3. Vérifier l'installation
   ```bash
   python -c "import deimos; print(deimos.__version__)"
   ```

**Prérequis**
- 🗂️ Fichiers Blancs
  - Placer les fichiers blancs dans `data/input/blanks/`
- ⚠️ Base de Données de Référence  
  - Base de données NORMAN 📥 [Télécharger ici](https://drive.google.com/file/d/1mZa1r9RZ4Ioy1cILJqIteAz3vUs_UIaU/view) 
  - Créer un dossier `databases` dans `data/input/`
  - Copier le fichier `norman_all_ccs_all_rt_pos_neg_with_ms2.h5` dans `data/input/databases/`

### 🐍💻 Environnement Python
1. Cloner le repository
   ```bash
   git clone https://github.com/narvall018/pipeline_identification_ms.git
   ```
2. Créer un environnement Python
   ```bash
   python3 -m venv <NOM>
   ```
3. Activer l'environnement
   - Sous Linux/macOS :
     ```bash  
     source <NOM>/bin/activate
     ```
   - Sous Windows :
     ```bash
     <NOM>\Scripts\activate  
     ```
4. Aller dans le dossier du repository
   ```bash
   cd pipeline_identification_ms/
   ```
5. Installer les dépendances
   ```bash
   pip3 install -r requirements.txt
   ```

**Prérequis**
- 🗂️ Fichiers Blancs
  - Placer les fichiers blancs dans `data/input/blanks/`
- ⚠️ Base de Données de Référence
  - Base de données NORMAN 📥 [Télécharger ici](https://drive.google.com/file/d/1mZa1r9RZ4Ioy1cILJqIteAz3vUs_UIaU/view)
  - Créer un dossier `databases` dans `data/input/`
  - Copier le fichier `norman_all_ccs_all_rt_pos_neg_with_ms2.h5` dans `data/input/databases/`

## 📁 Structure du Projet
```
pipeline_identification_ms/
├── data/
│   ├── input/  
│   │   ├── samples/          # Fichiers .parquet des échantillons
│   │   ├── blanks/           # Fichiers blancs
│   │   ├── calibration/      # Calibration CCS
│   │   └── databases/        # Base de données
│   ├── intermediate/         # Données intermédiaires  
│   │   └── samples/
│   │       └── nom_echantillon/
│   │           ├── common_peaks.parquet
│   │           └── ms1/
│   │               └── all.parquet
│   └── output/               # Résultats et visualisations
├── logs/
├── scripts/ 
│   ├── config/               # Configuration
│   ├── processing/           # Traitement
│   ├── utils/                # Utilitaires
│   └── visualization/        # Visualisation 
└── tests/
```

## 🚀 Utilisation
1. Placer les fichiers :
   - Fichiers échantillons `.parquet` dans `data/input/samples/`
   - Fichiers blancs dans `data/input/blanks/`
   - Fichiers de calibration dans `data/input/calibration/`
   - Base de données dans `data/input/databases/`
2. Exécuter :
   ```bash
   python main.py
   ```

## ⚙️ Configuration
Configuration dans `scripts/config/config.py` :

```python
# Tolérances d'identification
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

## ⚖️ Licence  
Ce projet est sous licence [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/). Vous êtes libre de :
- **Partager** : copier et redistribuer le matériel sous quelque support que ce soit ou sous n'importe quel format.
- **Adapter** : remixer, transformer et créer à partir du matériel.

Selon les conditions suivantes :
- **Attribution** : Vous devez donner le crédit approprié, fournir un lien vers la licence et indiquer si des modifications ont été apportées. Vous devez le faire de la manière suggérée par l'auteur, mais pas d'une manière qui suggère qu'il vous soutient ou soutient votre utilisation du matériel.
- **Utilisation non commerciale** : Vous ne pouvez pas utiliser le matériel à des fins commerciales.

En savoir plus sur la licence [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)

