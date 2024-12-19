import torch
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from skimage.segmentation import clear_border
import imutils

# Charger le modèle YOLO pour les petites cases
cell_model = YOLO("./models/best_boxes.pt")  # Modèle pour les petites cases

# Fonction de prétraitement des cellules pour détecter les chiffres
def extract_digit(cell, debug=False):
    # Convertir l'image de la cellule en niveaux de gris
    gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    
    # Appliquer un seuillage automatique à la cellule, puis nettoyer les bords
    thresh = cv2.threshold(gray_cell, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)
    
    # Visualisation du prétraitement (optionnel)
    if debug:
        cv2.imshow("Cell Thresh", thresh)
        cv2.waitKey(0)
    
    # Trouver les contours dans la cellule seuillée
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    # Si aucun contour n'est trouvé, c'est une cellule vide
    if len(cnts) == 0:
        return None
    
    # Sinon, trouver le plus grand contour et créer un masque pour ce contour
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)
    
    # Calculer le pourcentage de pixels masqués par rapport à l'aire totale de l'image
    (h, w) = thresh.shape
    percentFilled = cv2.countNonZero(mask) / float(w * h)
    
    # Si moins de 3% de la cellule est remplie, ignorer le contour (probablement du bruit)
    if percentFilled < 0.03:
        return None
    
    # Appliquer le masque à la cellule seuillée
    digit = cv2.bitwise_and(thresh, thresh, mask=mask)
    
    # Visualiser le résultat (optionnel)
    if debug:
        cv2.imshow("Digit", digit)
        cv2.waitKey(0)
    
    return digit


# Charger l'image
image_path = "./test3.jpg"
img = cv2.imread(image_path)

# Détection des petites cases avec le modèle YOLO `best_boxes`
cell_results = cell_model(image_path, conf=0.5)

# Extraire les boîtes de chaque case
boxes = []
for result in cell_results:
    for box in result.boxes:
        box_coords = box.xyxy[0].cpu().numpy().astype(int)  # Récupérer les coordonnées des boîtes
        boxes.append(box_coords)

# Fonction de reconnaissance des chiffres avec pytesseract
def recognize_digit(cell_image):
    # Extraire et prétraiter le chiffre dans la cellule
    preprocessed_cell = extract_digit(cell_image)
    
    if preprocessed_cell is None:
        return 0
    
    # Utiliser pytesseract pour extraire le texte
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=123456789'
    result = pytesseract.image_to_string(preprocessed_cell, config=custom_config)
    
    # Essayer de convertir la sortie en un chiffre entier
    try:
        return int(result.strip()) if result.strip().isdigit() else 0
    except:
        return 0

# Création de la grille Sudoku (9x9)
grid = np.zeros((9, 9), dtype=np.int32)

# Taille de chaque case dans l'image
height, width = img.shape[:2]
cell_height = height // 9
cell_width = width // 9

# Afficher les boîtes de détection sur l'image et lire les chiffres dans chaque case
for box in boxes:
    x1, y1, x2, y2 = box
    # Découper chaque petite case
    cell = img[y1:y2, x1:x2]
    
    # Afficher la détection de la case
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Rectangle bleu pour la détection
    
    # Lire le chiffre dans la case
    digit = recognize_digit(cell)
    
    # Calculer la position de la case dans la grille avec plus de précision
    row = int((y1 + y2) / 2 / cell_height)  # Utiliser le centre de la case pour plus de précision
    col = int((x1 + x2) / 2 / cell_width)  # Utiliser le centre de la case pour plus de précision
    
    # Mettre à jour la grille avec le chiffre reconnu
    if digit != 0:
        grid[row][col] = digit

# Afficher la grille complète
for i in range(9):
    for j in range(9):
        print(grid[i][j], end=" ")
    print()

# Afficher l'image avec les détections des boîtes et chiffres
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Détection des petites cases et chiffres")
plt.axis("off")
plt.show()
