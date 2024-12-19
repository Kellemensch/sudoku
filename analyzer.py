import os
import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract
import matplotlib.pyplot as plt
from skimage.segmentation import clear_border
import imutils

# Charger les modèles YOLO pour détecter la grille et les cases
grid_model = YOLO("./models/best_grid.pt")  # Modèle pour détecter la grande grille de Sudoku
cell_model = YOLO("./models/best_boxes.pt")  # Modèle pour détecter les petites cases

# Fonction de prétraitement pour extraire un chiffre d'une cellule
def extract_digit(cell, debug=False):
    # Convertir la cellule en niveaux de gris
    gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    # Appliquer un seuillage binaire inversé
    thresh = cv2.threshold(gray_cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # Supprimer les bordures superflues
    thresh = clear_border(thresh)
    
    if debug:
        # Afficher la cellule seuillée si le mode debug est activé
        cv2.imshow("Cell Thresh", thresh)
        cv2.waitKey(0)
    
    # Trouver les contours dans l'image seuillée
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    if len(cnts) == 0:
        # Aucun chiffre détecté dans la cellule
        return None
    
    # Extraire le plus grand contour (probable chiffre)
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)
    
    # Vérifier si la surface du contour est suffisante pour représenter un chiffre
    (h, w) = thresh.shape
    percentFilled = cv2.countNonZero(mask) / float(w * h)
    if percentFilled < 0.01:
        return None
    
    # Extraire le chiffre de la cellule
    digit = cv2.bitwise_and(thresh, thresh, mask=mask)
    return digit

# Fonction pour reconnaître un chiffre à partir d'une cellule d'image
def recognize_digit(cell_image, debug=False):
    # Prétraiter la cellule pour extraire le chiffre
    preprocessed_cell = extract_digit(cell_image, debug)
    
    if preprocessed_cell is not None:
        # Configurer Tesseract pour reconnaître uniquement les chiffres
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=123456789'
        result = pytesseract.image_to_string(preprocessed_cell, config=custom_config).strip()
        
        # Convertir le résultat en entier ou retourner 0 en cas d'erreur
        try:
            digit = int(result) if result.isdigit() else 0
        except ValueError:
            digit = 0
    else:
        digit = 0  # Aucun chiffre détecté dans la cellule
    
    return digit

# Fonction pour détecter la grande grille de Sudoku et extraire les chiffres
def detect_sudoku_grid(image_path, debug=False):
    # Charger l'image à analyser
    img = cv2.imread(image_path)
    
    # Détecter la grande grille avec YOLO
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
            # Afficher la grille découpée si le mode debug est activé
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))
            plt.title(f"Grille de Sudoku découpée: {image_path}")
            plt.axis("off")
            plt.show()
        
        # Détecter les petites cases avec YOLO
        cell_results = cell_model(img_cropped, conf=0.5)
        boxes = []
        for result in cell_results:
            for box in result.boxes:
                box_coords = box.xyxy[0].cpu().numpy().astype(int)
                boxes.append(box_coords)
        
        # Dimensions des cellules estimées
        height, width = img_cropped.shape[:2]
        cell_height = height // 9
        cell_width = width // 9
        
        # Initialiser une grille 9x9 avec des zéros
        grid = np.zeros((9, 9), dtype=np.int32)
        
        for box in boxes:
            # Extraire les coordonnées de chaque case détectée
            x1, y1, x2, y2 = box
            cell = img_cropped[y1:y2, x1:x2]
            
            # Reconnaître le chiffre dans la cellule
            digit = recognize_digit(cell)
            
            # Calculer la position de la cellule dans la grille
            row = int((y1 + y2) / 2 / cell_height)
            col = int((x1 + x2) / 2 / cell_width)
            
            # Insérer le chiffre dans la grille
            grid[row][col] = digit
        
        return grid
    else:
        # Retourner None si aucune grille n'est détectée
        return None
