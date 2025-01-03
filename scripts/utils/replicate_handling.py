#scripts/utils/replicate_handling.py
#-*- coding:utf-8 -*-

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional
from ..config.config import Config

class ReplicateHandler:
    """Classe pour gérer le regroupement et le traitement des fichiers réplicats."""
    
    def __init__(self):
        """Initialise le gestionnaire de réplicats avec la configuration."""
        self.logger = logging.getLogger(__name__)
        self.config = Config
        # Pattern pour capturer le nom de base
        self.base_pattern = r"(.+?)_replicate_\d+(?:_\d+)?$"

    def group_replicates(self, sample_files: List[Path]) -> Dict[str, List[Path]]:
        """
        Regroupe les fichiers de réplicats par échantillon.
        
        Args:
            sample_files: Liste des chemins des fichiers d'échantillons
            
        Returns:
            Dict[str, List[Path]]: Dictionnaire des réplicats par échantillon
        """
        try:
            replicate_groups = {}
            
            for file_path in sample_files:
                # Récupère le nom du fichier sans extension
                file_name = file_path.stem
                match = re.match(self.base_pattern, file_name)
                
                if match:
                    # Extrait le nom de base (tout ce qui est avant _replicate)
                    base_name = match.group(1)
                    
                    # Ajoute le fichier au groupe correspondant
                    if base_name not in replicate_groups:
                        replicate_groups[base_name] = []
                    replicate_groups[base_name].append(file_path)
            
            # Tri des réplicats dans chaque groupe
            for base_name in replicate_groups:
                replicate_groups[base_name].sort()
                self.logger.info(f"Groupe {base_name}: {len(replicate_groups[base_name])} réplicats trouvés")
            
            return replicate_groups

        except Exception as e:
            self.logger.error(f"Erreur lors du groupement des réplicats : {str(e)}")
            raise

    def validate_replicate_group(
        self,
        group_name: str,
        replicate_files: List[Path]
    ) -> bool:
        """
        Valide un groupe de réplicats.
        
        Args:
            group_name: Nom du groupe
            replicate_files: Liste des fichiers réplicats
            
        Returns:
            bool: True si le groupe est valide
        """
        try:
            # Vérifie le nombre minimum de réplicats
            min_replicates = Config.REPLICATE.min_replicates
            if len(replicate_files) < min_replicates:
                self.logger.warning(
                    f"Groupe {group_name} : nombre insuffisant de réplicats "
                    f"({len(replicate_files)}/{min_replicates})"
                )
                return False
            
            # Vérifie l'existence des fichiers
            for rep_file in replicate_files:
                if not rep_file.exists():
                    self.logger.error(f"Fichier manquant : {rep_file}")
                    return False
                
            return True

        except Exception as e:
            self.logger.error(f"Erreur lors de la validation du groupe {group_name} : {str(e)}")
            return False

    def get_replicate_number(self, file_path: Path) -> Optional[int]:
        """
        Extrait le numéro de réplicat d'un nom de fichier.
        
        Args:
            file_path: Chemin du fichier
            
        Returns:
            Optional[int]: Numéro de réplicat ou None si non trouvé
        """
        try:
            # Recherche du pattern _replicate_X
            match = re.search(r'_replicate_(\d+)', file_path.stem)
            if match:
                return int(match.group(1))
            return None

        except Exception as e:
            self.logger.error(f"Erreur lors de l'extraction du numéro de réplicat : {str(e)}")
            return None

    def check_replicate_sequence(
        self,
        group_name: str,
        replicate_files: List[Path]
    ) -> bool:
        """
        Vérifie que la séquence des numéros de réplicats est continue.
        
        Args:
            group_name: Nom du groupe
            replicate_files: Liste des fichiers réplicats
            
        Returns:
            bool: True si la séquence est continue
        """
        try:
            numbers = []
            for file_path in replicate_files:
                num = self.get_replicate_number(file_path)
                if num is not None:
                    numbers.append(num)
            
            numbers.sort()
            
            # Vérifie que la séquence est continue
            if numbers and numbers != list(range(min(numbers), max(numbers) + 1)):
                self.logger.warning(
                    f"Groupe {group_name} : séquence de réplicats discontinue : {numbers}"
                )
                return False
                
            return True

        except Exception as e:
            self.logger.error(f"Erreur lors de la vérification de la séquence : {str(e)}")
            return False

    def find_sample_replicates(
        self,
        sample_name: str,
        search_dir: Optional[Path] = None
    ) -> List[Path]:
        """
        Trouve tous les réplicats d'un échantillon donné.
        
        Args:
            sample_name: Nom de l'échantillon
            search_dir: Répertoire de recherche (optionnel)
            
        Returns:
            List[Path]: Liste des chemins des réplicats trouvés
        """
        try:
            if search_dir is None:
                search_dir = Config.PATHS.INPUT_SAMPLES
            
            pattern = f"{sample_name}_replicate_*"
            replicate_files = list(search_dir.glob(pattern))
            
            if not replicate_files:
                self.logger.warning(f"Aucun réplicat trouvé pour {sample_name}")
            else:
                self.logger.info(f"Trouvé {len(replicate_files)} réplicats pour {sample_name}")
            
            return sorted(replicate_files)

        except Exception as e:
            self.logger.error(f"Erreur lors de la recherche des réplicats : {str(e)}")
            return []