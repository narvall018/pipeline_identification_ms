# scripts/config/config.py

class Config:
    # Chemins des données
    INPUT_SAMPLES = "data/input/samples"
    INPUT_CALIBRANTS = "data/input/calibrants"
    INPUT_DATABASES = "data/input/databases"
    
    INTERMEDIATE_SAMPLES = "data/intermediate/samples"
    INTERMEDIATE_CALIBRANTS = "data/intermediate/calibrants"
    
    # Paramètres de détection des pics
    PEAK_DETECTION = {
        'threshold': 100,           # Seuil d'intensité
        'smooth_iterations': 7,     # Nombre d'itérations pour le lissage
        'smooth_radius': {          # Rayons de lissage pour chaque dimension
            'mz': 0,
            'drift_time': 1,
            'retention_time': 0
        },
        'peak_radius': {           # Rayons pour la détection des pics
            'mz': 2,
            'drift_time': 10,
            'retention_time': 0
        }
    }
    
    # Paramètres de clustering
    CLUSTERING = {
        'tolerances': {
            'mz': 1e-4,    # Tolérance m/z (en ppm ou Da)
            'dt': 0.10,    # Tolérance temps de dérive (en %)
            'rt': 0.20     # Tolérance temps de rétention (en %)
        },
        'dbscan': {
            'eps': 1.0,          # Epsilon pour DBSCAN
            'min_samples': 2     # Nombre minimum de points pour former un cluster
        }
    }
    
    
        # Paramètres d'identification
    IDENTIFICATION = {
        'database_file': "norman_all_ccs_all_rt_pos_neg_with_ms2.h5",
        'database_key': 'positive',
        # Tolérances
        'tolerances': {
            'mz_ppm': 5,      # 5 ppm pour m/z
            'ccs_percent': 8,   # 8% pour CCS
            'rt_min': 2        # 2 minutes pour RT
        },
        # Poids pour le score global
        'weights': {
            'mz': 0.4,
            'ccs': 0.4,
            'rt': 0.2
        }
    }
    
    # Colonnes de la base de données
    DB_COLUMNS = {
        'name': 'Name',
        'mz': 'mz',
        'ccs_exp': 'ccs_exp',
        'ccs_pred': 'ccs_pred',
        'rt_obs': 'Observed_RT',
        'rt_pred': 'Predicted_RT'
    }
    
    
    # Paramètres MS2
    MS2_EXTRACTION = {
        'rt_tolerance': 0.00422,  # minutes
        'dt_tolerance': 0.22,     # ms
        'mz_round_decimals': 3,
        'max_peaks': 10,
        'intensity_scale': 999    # Facteur de normalisation
    }
