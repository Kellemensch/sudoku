import os
import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract
import matplotlib.pyplot as plt
from skimage.segmentation import clear_border
import imutils

# Charger les modèles YOLO pour détecter la grille et les cases
grid_model = YOLO("./models/best_grid.pt")  # Modèle pour la grille de Sudoku
cell_model = YOLO("./models/best_boxes.pt")  # Modèle pour les cases individuelles

# Fonction de prétraitement pour extraire un chiffre d'une cellule
def extract_digit(cell, debug=False):
    """
    Prend une cellule d'image et tente d'extraire le chiffre.
    Applique un seuillage, supprime les bordures et isole les contours principaux.
    """
    # Convertir la cellule en niveaux de gris
    gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    # Appliquer un seuillage binaire inversé
    thresh = cv2.threshold(gray_cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # Nettoyer les bordures pour éviter les artefacts
    thresh = clear_border(thresh)
    
    if debug:
        cv2.imshow("Cell Thresholded", thresh)
        cv2.waitKey(0)
    
    # Trouver les contours de l'image seuillée
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    if len(cnts) == 0:
        # Retourner None si aucun contour significatif n'est trouvé
        return None
    
    # Extraire le plus grand contour
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)
    
    # Vérifier si le contour remplit suffisamment la cellule
    (h, w) = thresh.shape
    percentFilled = cv2.countNonZero(mask) / float(w * h)
    if percentFilled < 0.03:  # Ajuster ce seuil pour éviter les détections incorrectes
        return None
    
    # Extraire l'image du chiffre isolé
    digit = cv2.bitwise_and(thresh, thresh, mask=mask)
    return digit

# Fonction pour reconnaître un chiffre dans une cellule
def recognize_digit(cell_image, debug=False):
    """
    Reconnaît un chiffre dans une cellule d'image prétraitée.
    Utilise Tesseract OCR pour interpréter le chiffre extrait.
    """
    preprocessed_cell = extract_digit(cell_image, debug)
    
    if preprocessed_cell is not None:
        # Configurer Tesseract pour reconnaître uniquement les chiffres
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=123456789'
        result = pytesseract.image_to_string(preprocessed_cell, config=custom_config).strip()
        
        try:
            digit = int(result) if result.isdigit() else 0
        except ValueError:
            digit = 0  # Retourner 0 en cas d'erreur de reconnaissance
    else:
        digit = 0  # Aucun chiffre détecté
    
    return digit

# Fonction pour détecter la grande grille et extraire les chiffres
def detect_sudoku_grid(image_path, debug=False):
    """
    Détecte la grille de Sudoku et extrait les chiffres dans une structure 9x9.
    """
    img = cv2.imread(image_path)
    
    # Utiliser le modèle YOLO pour détecter la grille principale
    grid_results = grid_model(image_path, conf=0.5)
    grid_box = None
    for result in grid_results:
        for box in result.boxes:
            box_coords = box.xyxy[0].cpu().numpy().astype(int)
            grid_box = box_coords
    
    if grid_box is not None:
        # Découper l'image pour isoler la grille
        x1, y1, x2, y2 = grid_box
        img_cropped = img[y1:y2, x1:x2]
        
        if debug:
            # Afficher la grille détectée
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))
            plt.title("Grille de Sudoku découpée")
            plt.axis("off")
            plt.show()
        
        # Utiliser le second modèle YOLO pour détecter les cases
        cell_results = cell_model(img_cropped, conf=0.5)
        boxes = []
        for result in cell_results:
            for box in result.boxes:
                box_coords = box.xyxy[0].cpu().numpy().astype(int)
                boxes.append(box_coords)
        
        # Calculer la taille des cellules
        height, width = img_cropped.shape[:2]
        cell_height = height // 9
        cell_width = width // 9
        
        # Initialiser une grille 9x9
        grid = np.zeros((9, 9), dtype=np.int32)
        
        for box in boxes:
            x1, y1, x2, y2 = box
            cell = img_cropped[y1:y2, x1:x2]
            
            # Reconnaître le chiffre dans chaque cellule
            digit = recognize_digit(cell)
            
            # Déterminer la position de la cellule dans la grille
            row = int((y1 + y2) / 2 / cell_height)
            col = int((x1 + x2) / 2 / cell_width)
            
            # Insérer le chiffre dans la grille
            grid[row][col] = digit
        
        return grid_box, grid
    else:
        return None, None

# Fonction pour écrire la solution sur l'image originale
def write_solution_on_image(image_path, solution_grid, output_path):
    """
    Annoter l'image originale avec la solution du Sudoku.
    """
    img = cv2.imread(image_path)
    
    # Détecter la grille pour obtenir ses coordonnées
    grid_box, _ = detect_sudoku_grid(image_path)
    
    if grid_box is None:
        print("Impossible de détecter la grille pour écrire la solution.")
        return
    
    # Découper l'image pour travailler sur la zone de la grille
    x1, y1, x2, y2 = grid_box
    img_cropped = img[y1:y2, x1:x2]
    
    # Calculer la taille des cellules
    height, width = img_cropped.shape[:2]
    cell_height = height // 9
    cell_width = width // 9
    
    # Parcourir la grille pour annoter chaque cellule
    for i in range(9):
        for j in range(9):
            digit = solution_grid[i][j]
            if digit != 0:  # Éviter d'écrire les cellules vides
                x = j * cell_width + cell_width // 4
                y = i * cell_height + 3 * cell_height // 4
                cv2.putText(
                    img_cropped, str(digit), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )
    
    # Réintégrer la grille annotée dans l'image complète
    img[y1:y2, x1:x2] = img_cropped
    cv2.imwrite(output_path, img)
    print(f"Image annotée sauvegardée à : {output_path}")
