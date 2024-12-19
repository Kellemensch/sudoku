import argparse
import os
from train_model import train_model  # Importer la fonction d'entraînement
from solver import solve_grid  # Importer le solveur
from analyzer import detect_sudoku_grid  # Importer les fonctions d'analyse
from test_sudoku import run_tests
from afficheur import write_solution_on_image

global debug

def main():
    # Créer un parseur d'arguments
    parser = argparse.ArgumentParser(description="Analyseur de Sudoku, tests et entraînement de modèles")
    
    # Ajouter des arguments pour l'entraînement, les tests et l'analyse
    parser.add_argument("--train", action="store_true", help="Entraîner les modèles")
    parser.add_argument("--image", type=str, help="Chemin vers l'image à analyser")
    parser.add_argument("--truth", type=str, help="Chemin vers le fichier de vérité pour les tests")
    parser.add_argument("--test", action="store_true", help="Exécuter les tests sur un dossier d'images")
    parser.add_argument("--images_dir", type=str, help="Dossier contenant les images pour les tests (nécessaire avec --test)")
    parser.add_argument("--debug", action="store_true", help="Option de debug avec affichages")
    
    args = parser.parse_args()

    if args.debug:
        # Debug activé
        print("OPtion de debug activée...\n")
        debug = True
    else:
        debug = False

    if args.train:
        # Entraîner les modèles
        print("Entraînement des modèles...")
        train_model()

    elif args.image:
        # Analyser une image donnée
        if not os.path.exists(args.image):
            print(f"Erreur : L'image spécifiée {args.image} n'existe pas.")
            return
        print(f"Analyse de l'image : {args.image}")
        grid = detect_sudoku_grid(args.image, debug)
        if grid is not None:
            print("Grille détectée :")
            for row in grid:
                print(" ".join(map(str, row)))
            solution = solve_grid(grid)
            print("Solution de la grille :")
            if solution is not None:
                for row in solution:
                    print(" ".join(map(str, row)))
                write_solution_on_image(args.image, solution, "./solution.png")
            else:
                print("\033[91mPas de solution trouvée\033[0m\n")
        else:
            print("Impossible de détecter la grille.")

    elif args.test:
        # Exécuter les tests sur un dossier d'images
        # if not args.images_dir or not os.path.exists(args.images_dir):
        #     print("Erreur : Vous devez spécifier un dossier d'images existant avec --images_dir.")
        #     return
        # if not args.truth or not os.path.exists(args.truth):
        #     print("Erreur : Vous devez spécifier un fichier de réponses manuelles avec --truth.")
        #     return
        print("Exécution des tests...")
        accuracy = run_tests("./test_images/", "./test_images/truth_answers.txt", debug)
        print(f"Précision du modèle sur les tests : {accuracy:.2f}%")

    else:
        print("Veuillez spécifier une option valide : --train, --image, ou --test.")

if __name__ == "__main__":
    main()
