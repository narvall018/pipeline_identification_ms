
suppressPackageStartupMessages({
  library(data.table)
  library(Spectra)
  library(MsCoreUtils)
  library(dplyr)
  library(arrow)
  library(rhdf5)
})

read_db_spectra <- function(h5_path) {
    cat("\n📚 Chargement de la base de données MS2...\n")
    
    # Lire toutes les données de la clé positive
    h5closeAll()  # Fermer toutes les connexions existantes
    db <- h5read(h5_path, "positive")
    
    cat(f"   ✓ Nombre de composés dans la BDD: {length(db$peaks_ms2_mz)}\n")
    
    # Créer le DataFrame Spectra
    db_spectra <- DataFrame(
        msLevel = rep(2L, length(db$Name)),
        polarity = rep(1L, length(db$Name)),
        id = db$molecule_id,
        name = db$Name,
        mz = db$peaks_ms2_mz,
        intensity = db$peaks_ms2_intensities
    )
    
    cat("   ✓ Données MS2 chargées avec succès\n")
    Spectra(db_spectra)
}

process_matches_file <- function(file_path, db_spectra) {
    cat(sprintf("\n🔍 Traitement de %s\n", basename(file_path)))
    
    # Lecture du fichier de matches
    matches_df <- read_parquet(file_path)
    
    # Filtrage des matches avec MS2
    matches_ms2 <- matches_df[matches_df$has_ms2_db == 1, ]
    
    if (nrow(matches_ms2) == 0) {
        cat("⚠️ Aucun match avec MS2 trouvé\n")
        return(matches_df)
    }
    
    cat(sprintf("   ✓ Matches avec MS2: %d\n", nrow(matches_ms2)))
    
    # Préparation des spectres expérimentaux
    matches_spectra <- DataFrame(
        msLevel = rep(2L, nrow(matches_ms2)),
        polarity = rep(1L, nrow(matches_ms2)),
        id = matches_ms2$molecule_id,
        name = matches_ms2$match_name,
        mz = matches_ms2$peaks_mz_ms2,
        intensity = matches_ms2$peaks_intensities_ms2
    )
    matches_good <- Spectra(matches_spectra)
    
    # Calcul des scores MS2
    cat("   ⚡ Calcul des scores MS2...\n")
    ms2_scores <- numeric(nrow(matches_ms2))
    
    for (i in seq_len(nrow(matches_ms2))) {
        molecule_id <- matches_ms2$molecule_id[i]
        compound_search <- matches_good[i]
        compound_db <- db_spectra[db_spectra$id == molecule_id]
        
        if (length(compound_db) > 0) {
            sims <- compareSpectra(compound_search, compound_db, 
                                FUN = ndotproduct, ppm = 50)
            ms2_scores[i] <- max(sims) + 0.1
        } else {
            ms2_scores[i] <- NA
        }
        
        if (i %% 50 == 0) {
            cat(sprintf("      Progress: %d/%d\n", i, nrow(matches_ms2)))
        }
    }
    
    # Mise à jour du DataFrame
    matches_df$ms2_score <- 0
    matches_df$ms2_score[matches_df$has_ms2_db == 1] <- ms2_scores
    
    # Tri des résultats
    matches_df <- matches_df %>%
        arrange(desc(ms2_score), desc(global_score))
    
    # Sauvegarde
    write_parquet(matches_df, file_path)
    cat(sprintf("   ✓ Résultats sauvegardés: %s\n", basename(file_path)))
    
    return(matches_df)
}

main <- function() {
    cat("\n🚀 DÉMARRAGE DE LA COMPARAISON DES SPECTRES MS2\n")
    cat("================================================\n")
    
    # Chargement BDD
    base_dir <- normalizePath(".")
    db_path <- file.path(base_dir, "data/input/databases/norman_all_ccs_all_rt_pos_neg_with_ms2.h5")
    
    h5closeAll()  # S'assurer qu'il n'y a pas de connexions h5 ouvertes
    
    # Lecture spectres de référence
    db_spectra <- read_db_spectra(db_path)
    
    # Recherche des fichiers all_matches.parquet
    all_matches_files <- list.files(
        path = file.path(base_dir, "data/intermediate"),
        pattern = "all_matches\\.parquet$",
        recursive = TRUE,
        full.names = TRUE
    )
    
    if (length(all_matches_files) == 0) {
        stop("❌ Aucun fichier all_matches.parquet trouvé")
    }
    
    cat(sprintf("\n📁 Fichiers à traiter: %d\n", length(all_matches_files)))
    
    # Traitement des fichiers
    for (file_path in all_matches_files) {
        process_matches_file(file_path, db_spectra)
    }
    
    cat("\n✅ COMPARAISON MS2 TERMINÉE\n")
    cat("================================================\n")
}

# Exécution
main()