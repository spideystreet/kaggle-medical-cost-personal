 # Sauvegardez ce code dans un fichier merge_notebooks.py
import nbformat
import sys
import os

def merge_notebooks(output_path, input_paths):
    """
    Fusionne plusieurs notebooks Jupyter en un seul.
    
    Args:
        output_path: Chemin vers le nouveau notebook à créer
        input_paths: Liste des chemins vers les notebooks à fusionner
    """
    # Créer un nouveau notebook
    merged = nbformat.v4.new_notebook()
    merged.cells = []
    
    # Pour chaque notebook source
    for input_path in input_paths:
        # Ajouter un titre avec le nom du notebook
        notebook_name = os.path.basename(input_path).replace('.ipynb', '')
        title_cell = nbformat.v4.new_markdown_cell(f"# {notebook_name}")
        merged.cells.append(title_cell)
        
        # Lire le notebook source
        with open(input_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Ajouter toutes les cellules au nouveau notebook
        merged.cells.extend(nb.cells)
        
        # Ajouter une cellule de séparation
        merged.cells.append(nbformat.v4.new_markdown_cell("---"))
    
    # Sauvegarder le nouveau notebook
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(merged, f)
    
    print(f"Notebooks fusionnés dans {output_path}")

if __name__ == "__main__":
    notebooks = [
        "notebooks/01_eda.ipynb", 
        "notebooks/02_dataPreprocessing.ipynb", 
        "notebooks/03_training.ipynb",
        "notebooks/04_predictions.ipynb"
    ]
    
    merge_notebooks("notebooks/complete_analysis.ipynb", notebooks)