#!/usr/bin/env Rscript

# Supprimer tous les messages et avertissements
options(warn = -1)
options(dplyr.show_progress = FALSE)

# Rediriger les sorties vers null
log_null <- file("/dev/null", open = "wt")
sink(log_null, type = "output")
sink(log_null, type = "message")

# Charger les packages silencieusement
suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(MS2Quant)
  library(arrow)
  library(stringr)
  library(ggplot2)
})

# Restaurer la sortie standard
sink(type = "output")
sink(type = "message")
close(log_null)

# Chemins
base_dir <- "data/input/calibrants"
output_dir <- "output/quantification"
compounds_summary_path <- file.path(output_dir, "compounds_summary.csv")
features_path <- file.path("output/feature_matrix/features_complete.parquet")
path_eluent_file <- file.path(base_dir, "eluents/eluent_leesu.csv")
calib_samples_path <- file.path(base_dir, "samples/calibrants_samples.csv")

# Créer les sous-dossiers
sample_results_dir <- file.path(output_dir, "samples_quantification")
model_info_dir <- file.path(output_dir, "model_info")
plots_dir <- file.path(output_dir, "plots")
dir.create(sample_results_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(model_info_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)

# Charger les données silencieusement
calibrants <- suppressWarnings(fread(compounds_summary_path, showProgress = FALSE))
features <- suppressWarnings(read_parquet(features_path))
calib_samples <- suppressWarnings(fread(calib_samples_path, showProgress = FALSE))
calibrant_names <- unique(calib_samples$Name)

# Préparer les données de calibration
calibrants_adapted <- suppressWarnings(
  calibrants %>%
    mutate(
      identifier = Compound,
      area = Intensity,
      retention_time = RT
    ) %>%
    group_by(identifier) %>%
    filter(n() == 5) %>%
    ungroup()
)

# Obtenir la liste des échantillons à traiter
samples_to_process <- suppressWarnings(
  features %>%
    filter(confidence_level == 1) %>%
    pull(samples) %>%
    str_split(",") %>%
    unlist() %>%
    str_trim() %>%
    unique() %>%
    setdiff(calibrant_names)
)

# Fonction pour sauvegarder les informations du modèle
save_model_info <- function(model_summary) {
  capture.output(
    print(model_summary),
    file = file.path(model_info_dir, "calibration_model_summary.txt")
  )
}

# Traiter chaque échantillon
for(sample in samples_to_process) {
  tryCatch({
    # Rediriger les sorties vers null pour chaque échantillon
    temp_log <- file("/dev/null", open = "wt")
    sink(temp_log, type = "output")
    sink(temp_log, type = "message")
    
  identification_samples <- features %>%
  filter(
    confidence_level == 1,
    str_detect(samples, sample)
  ) %>%
  mutate(
    identifier = match_name,
    SMILES = match_smiles,
    retention_time = retention_time,
    area = intensity,
    conc_M = NA,
    daphnia_LC50 = daphnia_LC50_48_hr_ug/L,
    algae_EC50 = algae_EC50_72_hr_ug/L,
    pimephales_LC50 = pimephales_LC50_96_hr_ug/L
  ) %>%
  select(
    identifier, SMILES, retention_time, area, conc_M,
    daphnia_LC50, algae_EC50, pimephales_LC50
  ) %>%
  distinct()

  # Combiner avec les calibrants
  data_combined <- bind_rows(
    calibrants_adapted %>%
      select(identifier, SMILES, retention_time, area, conc_M,
             daphnia_LC50, algae_EC50, pimephales_LC50),
    identification_samples
  )
    
    # Exécuter MS2Quant
    MS2Quant_results <- MS2Quant_quantify(data_combined,
                                         path_eluent_file,
                                         organic_modifier = "MeCN",
                                         pH_aq = 2.7)
    
    # Restaurer la sortie standard
    sink(type = "output")
    sink(type = "message")
    close(temp_log)
    
    # Sauvegarder les résultats
    write.csv(MS2Quant_results$suspects_concentrations,
              file.path(sample_results_dir, paste0(sample, "_quantification.csv")),
              row.names = FALSE)
    
    # Sauvegarder les informations du modèle (une seule fois)
    if (!file.exists(file.path(model_info_dir, "calibration_model_summary.txt")) &&
        !is.null(MS2Quant_results$calibration_linear_model_summary)) {
      save_model_info(MS2Quant_results$calibration_linear_model_summary)
    }
    
    # Sauvegarder le plot (une seule fois)
    if (!file.exists(file.path(plots_dir, "calibrants_plot.png")) &&
        !is.null(MS2Quant_results$calibrants_separate_plots)) {
      ggsave(
        file.path(plots_dir, "calibrants_plot.png"), 
        plot = MS2Quant_results$calibrants_separate_plots,
        width = 8, 
        height = 6, 
        dpi = 300
      )
    }
    
    cat("✅ Résultats sauvegardés pour", sample, "\n")
  }, error = function(e) {
    sink(type = "output")
    sink(type = "message")
    cat("❌ Erreur pour l'échantillon", sample, ":", e$message, "\n")
  })
}

cat("\n✅ Tous les résultats ont été sauvegardés dans", output_dir, "\n")
cat("  • Quantification:", sample_results_dir, "\n")
cat("  • Informations des modèles:", model_info_dir, "\n")
cat("  • Plots de calibration:", plots_dir, "\n")