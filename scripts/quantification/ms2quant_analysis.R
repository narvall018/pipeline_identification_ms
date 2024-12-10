#!/usr/bin/env Rscript

# Supprimer tous les messages et avertissements
options(warn = -1)
options(dplyr.show_progress = FALSE)
options(readr.show_progress = FALSE)
Sys.setenv(XGBOOST_WARNING = 0)

# Fonction pour calculer la masse molaire selon l'adduit
get_adduct_mass <- function(adduct) {
  switch(adduct,
         "[M+H]+" = 1.007825,
         "[M+Na]+" = 22.989769,
         "[M+NH4]+" = 18.034374,
         1.007825)  # Défaut pour [M+H]+
}

# Fonction pour calculer la concentration en g/L
calculate_concentration <- function(molar_concentration, mz, adduct) {
  adduct_mass <- get_adduct_mass(adduct)
  molecular_mass <- mz - adduct_mass
  return(molar_concentration * molecular_mass)
}

# Fonction pour gérer proprement la redirection des sorties
redirect_output <- function(expr) {
  temp_log <- file("/dev/null", open = "wt")
  sink(temp_log, type = "message")
  sink(temp_log, type = "output")
  
  result <- tryCatch({
    expr
  }, finally = {
    suppressWarnings({
      sink(type = "message", NULL)
      sink(type = "output", NULL)
      close(temp_log)
    })
  })
  
  return(result)
}

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(MS2Quant)
  library(stringr)
  library(ggplot2)
})

# Chemins des fichiers et dossiers
output_root <- "output/quantification"
data_summary_path <- file.path(output_root, "compounds_summary.csv")
eluent_file_path <- "data/input/calibrants/eluents/eluent_leesu.csv"
results_dir <- file.path(output_root, "samples_quantification")
model_dir <- file.path(output_root, "model_info")
plots_dir <- file.path(output_root, "calibration_plots")

# Création des dossiers
invisible(sapply(c(results_dir, model_dir, plots_dir), 
                dir.create, recursive = TRUE, showWarnings = FALSE))

# Lecture des données
compounds_data <- suppressWarnings(fread(data_summary_path, showProgress = FALSE))

# Identification des calibrants valides (5 points)
calibration_data <- compounds_data[compounds_data$Is_Calibration == TRUE, ]
compounds_with_5points <- names(table(calibration_data$Compound)[table(calibration_data$Compound) == 5])
valid_calibration_data <- calibration_data[calibration_data$Compound %in% compounds_with_5points, ]

# Colonnes à conserver
selected_columns <- c(
  "identifier", "conc_M", "conc", "logRF_pred", "area", "mz", "RT", "DT", "CCS",
  "Adduct", "SMILES", "Feature_ID", "Confidence_Level", 
  "daphnia_LC50_48_hr_ug/L", "algae_EC50_72_hr_ug/L", 
  "pimephales_LC50_96_hr_ug/L", "Sample"
)

# Liste des échantillons à traiter
samples_list <- unique(compounds_data[compounds_data$Is_Calibration == FALSE,]$Sample)
model_is_saved <- FALSE
quantification_results <- list()

for(current_sample in samples_list) {
  cat(sprintf("Traitement de %s...\n", current_sample))
  
  # Données de l'échantillon courant
  sample_data <- compounds_data[compounds_data$Is_Calibration == FALSE & 
                               compounds_data$Sample == current_sample, ]
  
  # Préparation des données pour MS2Quant
  ms2quant_data <- rbind(valid_calibration_data, sample_data)
  ms2quant_data$identifier <- ms2quant_data$Compound
  ms2quant_data$area <- ms2quant_data$Intensity
  ms2quant_data$retention_time <- ms2quant_data$RT
  
  # Exécution de MS2Quant avec redirection des sorties
  ms2quant_output <- redirect_output({
    tryCatch({
      MS2Quant_quantify(
        ms2quant_data,
        eluent_file_path,
        organic_modifier = "MeCN",
        pH_aq = 2.7
      )
    }, error = function(e) {
      cat(sprintf("❌ Erreur MS2Quant: %s\n", e$message))
      return(NULL)
    })
  })
  
  if(!is.null(ms2quant_output)) {
    # Sauvegarde du modèle (une seule fois)
    if(!model_is_saved && !is.null(ms2quant_output$calibration_linear_model_summary)) {
      capture.output(
        print(ms2quant_output$calibration_linear_model_summary),
        file = file.path(model_dir, "calibration_model_summary.txt")
      )
      
      if(!is.null(ms2quant_output$calibrants_separate_plots)) {
        ggsave(
          file.path(plots_dir, "calibration_plots.png"),
          ms2quant_output$calibrants_separate_plots,
          width = 10,
          height = 8,
          dpi = 300
        )
      }
      model_is_saved <- TRUE
      cat("✓ Modèle et plots sauvegardés\n")
    }
    
    if(!is.null(ms2quant_output$suspects_concentrations)) {
      # Préparation des résultats
      quant_results <- as.data.frame(ms2quant_output$suspects_concentrations)
      
      # Calcul des concentrations
      quant_results$conc <- mapply(
        calculate_concentration,
        quant_results$conc_M,
        sample_data$mz,
        sample_data$Adduct
      )
      
      # Combinaison et sélection des colonnes
      final_results <- cbind(quant_results, as.data.frame(sample_data))
      final_results <- final_results[, intersect(names(final_results), selected_columns)]
      
      # Sauvegarde
      output_file <- file.path(results_dir, paste0(current_sample, "_quantification.csv"))
      fwrite(as.data.table(final_results), output_file, showProgress = FALSE)
      cat(sprintf("✓ Résultats sauvegardés: %s\n", output_file))
      
      quantification_results[[current_sample]] <- final_results
    }
  }
}

# Compilation des résultats
if(length(quantification_results) > 0) {
  all_results <- suppressWarnings(rbindlist(lapply(quantification_results, as.data.table), fill = TRUE))
  summary_file <- file.path(output_root, "all_quantification_results.csv")
  fwrite(all_results, summary_file, showProgress = FALSE)
  cat(sprintf("\n✓ Résultats combinés sauvegardés: %s\n", summary_file))
}

cat("\n✅ Traitement terminé\n")
cat(sprintf("• Résultats de quantification: %s\n", results_dir))
cat(sprintf("• Résumé du modèle: %s\n", model_dir))
cat(sprintf("• Plots de calibration: %s\n", plots_dir))
cat(sprintf("• Fichier combiné: %s\n", "all_quantification_results.csv"))