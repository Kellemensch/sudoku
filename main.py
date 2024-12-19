import argparse
import os
from train_model import train_model  # Fonction pour entraîner les modèles
from solver import solve_grid  # Fonction pour résoudre une grille de Sudoku
from analyzer import detect_sudoku_grid  # Fonction pour détecter la grille dans une image
from test_sudoku import run_tests  # Fonction pour exécuter les tests
from afficheur import write_solution_on_image  # Fonction pour annoter une image avec la solution

# Variable globale pour activer/désactiver le mode debug
global debug

def main():
    # Créer un parseur d'arguments pour les options de ligne de commande
    parser = argparse.ArgumentParser(description="Analyseur de Sudoku, tests et entraînement de modèles")
    
    # Définir les arguments disponibles
    parser.add_argument("--train", action="store_true", help="Entraîner les modèles")
    parser.add_argument("--image", type=str, help="Chemin vers l'image à analyser")
    parser.add_argument("--truth", type=str, help="Chemin vers le fichier de vérité pour les tests")
    parser.add_argument("--test", action="store_true", help="Exécuter les tests sur un dossier d'images")
    parser.add_argument("--debug", action="store_true", help="Activer le mode debug avec affichages supplémentaires")
    
    args = parser.parse_args()

    if args.debug:
        # Activer le mode debug
        print("Option de debug activée...\n")
        global debug
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
        grid = detect_sudoku_grid(args.image, debug)  # Détecter la grille de Sudoku
        if grid is not None:
            print("Grille détectée :")
            for row in grid:
                print(" ".join(map(str, row)))  # Afficher la grille détectée
            
            solution = solve_grid(grid)  # Résoudre la grille
            print("Solution de la grille :")
            if solution is not None:
                for row in solution:
                    print(" ".join(map(str, row)))  # Afficher la solution
                # Annoter l'image originale avec la solution
                write_solution_on_image(args.image, solution, "./solutions/solution.png")
            else:
                print("\033[91mPas de solution trouvée\033[0m\n")  # Afficher un message en rouge
        else:
            print("Impossible de détecter la grille.")

    elif args.test:
        # Exécuter les tests sur un ensemble d'images
        print("Exécution des tests...")
        if not args.images_dir or not os.path.exists(args.images_dir):
            print("Erreur : Vous devez spécifier un dossier d'images existant avec --images_dir.")
            return
        if not args.truth or not os.path.exists(args.truth):
            print("Erreur : Vous devez spécifier un fichier de réponses manuelles avec --truth.")
            return
        
        # Lancer les tests en utilisant les images et le fichier de réponses
        run_tests(args.images_dir, args.truth, debug)

    else:
        # Aucun argument valide fourni
        print("Veuillez spécifier une option valide : --train, --image, ou --test.")

if __name__ == "__main__":
    main()
