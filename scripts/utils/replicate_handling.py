from pathlib import Path
from typing import Dict, List
import re

def group_replicates(sample_files: List[Path]) -> Dict[str, List[Path]]:
    """
    Regroupe les fichiers de réplicats par échantillon.
    
    Args:
        sample_files: Liste des chemins des fichiers d'échantillons
        
    Returns:
        Dict[str, List[Path]]: Dictionnaire avec les noms d'échantillons comme clés
                             et la liste des chemins des réplicats comme valeurs
    """
    replicate_groups = {}
    
    # Pattern pour capturer le nom de base
    pattern = r"(.+?)_replicate_\d+(?:_\d+)?$"
    
    for file_path in sample_files:
        # Récupère le nom du fichier sans extension
        file_name = file_path.stem
        match = re.match(pattern, file_name)
        
        if match:
            # Extrait le nom de base (tout ce qui est avant _replicate)
            base_name = match.group(1)
            
            # Ajoute le fichier au groupe correspondant
            if base_name not in replicate_groups:
                replicate_groups[base_name] = []
            replicate_groups[base_name].append(file_path)
    
    # Trie les réplicats dans chaque groupe
    for base_name in replicate_groups:
        replicate_groups[base_name].sort()
    
    return replicate_groups
