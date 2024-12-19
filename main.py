import torch
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from skimage.segmentation import clear_border
import imutils

# Charger le modèle YOLO pour la grande grille
grid_model = YOLO("./models/best_grid.pt")  # Modèle pour la grande grille

# Charger le modèle YOLO pour les petites cases
cell_model = YOLO("./models/best_boxes.pt")  # Modèle pour les petites cases

# Fonction de prétraitement de l'image pour améliorer la reconnaissance
def extract_digit(cell, debug=False):
    # Convertir l'image en niveaux de gris
    gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    
    # Appliquer un seuillage automatique à la cellule et enlever les bords
    thresh = cv2.threshold(gray_cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)
    
    # Si débogage activé, afficher l'image après seuillage
    if debug:
        cv2.imshow("Cell Thresh", thresh)
        cv2.waitKey(0)
    
    # Trouver les contours dans la cellule
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    # Si aucun contour n'est trouvé, la cellule est vide
    if len(cnts) == 0:
        return None
    
    # Sinon, prendre le plus grand contour
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)
    
    # Calculer le pourcentage de pixels remplis par rapport à l'aire totale
    (h, w) = thresh.shape
    percentFilled = cv2.countNonZero(mask) / float(w * h)
    
    # Si moins de 2% de la cellule est remplie, on considère cela comme du bruit
    if percentFilled < 0.03:
        return None
    
    # Appliquer le masque à l'image seuillée
    digit = cv2.bitwise_and(thresh, thresh, mask=mask)
    
    # Si débogage activé, afficher l'image après le masquage
    if debug:
        cv2.imshow("Digit", digit)
        cv2.waitKey(0)
    
    return digit

# Fonction pour reconnaître un chiffre dans une cellule
def recognize_digit(cell_image, debug=False):
    # Appliquer le prétraitement à l'image de la cellule
    preprocessed_cell = extract_digit(cell_image, debug)
    
    # Si prétraitement réussi, utiliser pytesseract pour extraire le chiffre
    if preprocessed_cell is not None:
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=123456789'
        result = pytesseract.image_to_string(preprocessed_cell, config=custom_config).strip()
        
        # Convertir le résultat en entier (si possible)
        try:
            digit = int(result) if result.isdigit() else 0
        except ValueError:
            digit = 0
    else:
        digit = 0  # Si l'extraction échoue, retourner 0
    
    return digit

# Charger l'image
image_path = "./test.jpg"
img = cv2.imread(image_path)

# Détection de la grande grille avec le modèle YOLO `best_grid`
grid_results = grid_model(image_path, conf=0.5)

# Extraire la boîte de la grande grille (supposée être un seul rectangle)
grid_box = None
for result in grid_results:
    for box in result.boxes:
        box_coords = box.xyxy[0].cpu().numpy().astype(int)  # Récupérer les coordonnées de la grande grille
        grid_box = box_coords

# Recadrer l'image selon la grande grille détectée
if grid_box is not None:
    x1, y1, x2, y2 = grid_box
    img_cropped = img[y1:y2, x1:x2]
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))
    plt.title("Grille de Sudoku découpée")
    plt.axis("off")
    plt.show()

    # Détection des petites cases à l'intérieur de la grille recadrée avec le modèle YOLO `best_boxes`
    cell_results = cell_model(img_cropped, conf=0.5)

    # Extraire les boîtes des petites cases
    boxes = []
    for result in cell_results:
        for box in result.boxes:
            box_coords = box.xyxy[0].cpu().numpy().astype(int)  # Récupérer les coordonnées des boîtes
            boxes.append(box_coords)

    # Calculer les dimensions de la grille et de chaque case dans l'image recadrée
    height, width = img_cropped.shape[:2]
    cell_height = height // 9
    cell_width = width // 9

    # Création de la grille 9x9 pour afficher les chiffres
    grid = np.zeros((9, 9), dtype=np.int32)

    # Afficher les boîtes de détection et extraire les chiffres
    for box in boxes:
        x1, y1, x2, y2 = box
        # Découper chaque petite case
        cell = img_cropped[y1:y2, x1:x2]
        
        # Lire le chiffre dans la cellule
        digit = recognize_digit(cell)
        
        # Calculer la position de la case dans la grille
        row = int((y1 + y2) / 2 / cell_height)  # Utiliser le centre de la case pour plus de précision
        col = int((x1 + x2) / 2 / cell_width)  # Utiliser le centre de la case pour plus de précision
        
        # Mettre à jour la grille avec le chiffre reconnu
        grid[row][col] = digit

    # Afficher la grille complète
    for i in range(9):
        for j in range(9):
            print(grid[i][j], end=" ")
        print()

    # Afficher l'image recadrée avec les détections des boîtes et chiffres
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))
    plt.title("Détection des petites cases et chiffres dans la grille")
    plt.axis("off")
    plt.show()

else:
    print("Grille de Sudoku non détectée.")


def is_valid(grid, row, col, num):
    # Vérifier la ligne
    for c in range(9):
        if grid[row][c] == num:
            return False
    
    # Vérifier la colonne
    for r in range(9):
        if grid[r][col] == num:
            return False
    
    # Vérifier le sous-grille 3x3
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for r in range(start_row, start_row + 3):
        for c in range(start_col, start_col + 3):
            if grid[r][c] == num:
                return False
    
    return True

def solve(grid):
    # Chercher une cellule vide (représentée par 0)
    for row in range(9):
        for col in range(9):
            if grid[row][col] == 0:
                # Essayer les chiffres de 1 à 9
                for num in range(1, 10):
                    if is_valid(grid, row, col, num):
                        grid[row][col] = num
                        
                        # Récurse pour essayer de remplir la grille
                        if solve(grid):
                            return True
                        
                        # Si ça ne marche pas, annuler le choix
                        grid[row][col] = 0
                
                return False  # Retourner False si aucun chiffre n'est valide
    return True  # Retourner True si la grille est complète

if solve(grid):
    print("Grille résolue :")
    for i in range(9):
        for j in range(9):
            print(grid[i][j], end=" ")
        print()
else:
    print("Aucune solution trouvée.")
