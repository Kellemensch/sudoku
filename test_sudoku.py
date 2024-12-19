import os
import json
import numpy as np  # Importer NumPy pour la comparaison de matrices
from analyzer import detect_sudoku_grid
from solver import solve_grid

def run_tests(images_dir, truth_file, debug=False):
    # Charger les réponses correctes
    with open(truth_file, "r") as f:
        truth_answers = json.load(f)

    total_tests = 0
    total_accuracy = 0

    # Parcourir les images du dossier
    for image_name in os.listdir(images_dir):
        image_path = os.path.join(images_dir, image_name)
        if not image_name.endswith((".jpg", ".png", ".jpeg")):
            continue

        total_tests += 1
        print("---------------------------------")
        print(f"Analyse de {image_name}...\n")

        # Analyser l'image pour obtenir la grille
        detected_grid = detect_sudoku_grid(image_path, debug)
        if detected_grid is None:
            print("Échec de la détection.")
            continue

        
        true_solution = truth_answers.get(image_name)

        if true_solution is not None:
            true_solution = np.array(true_solution)
            detected_comp = np.array(detected_grid)
            
            # Comparer les grilles et calculer le taux de bonnes réponses
            correct_cells = np.sum(true_solution == detected_comp)
            total_cells = true_solution.size
            grid_accuracy = (correct_cells / total_cells) * 100

            if grid_accuracy == 100:
                color = "\033[92m"  # Vert pour 100%
            elif grid_accuracy >= 80:
                color = "\033[93m"  # Jaune pour 80%-99%
            else:
                color = "\033[91m"  # Rouge pour <80%

            # Résoudre la grille
            solved_grid = np.copy(detected_grid)
            solved_grid = solve_grid(solved_grid)
            print(f"Grille détectée : \n{detected_grid}\nSolution détectée : \n{solved_grid}\nGrille correcte : \n{true_solution}")
            print(f"{color}Taux de bonnes réponses pour {image_name}: {grid_accuracy:.2f}%\033[0m\n")
            total_accuracy += grid_accuracy
        else:
            print(f"Aucune réponse correcte trouvée pour {image_name} dans le fichier truth_answers.")

    # Calcul de la précision moyenne sur toutes les grilles
    average_accuracy = (total_accuracy / total_tests) if total_tests > 0 else 0
    print(f"\033[94mPrécision moyenne sur toutes les grilles: {average_accuracy:.2f}%\033[0m\n")
    return average_accuracy
