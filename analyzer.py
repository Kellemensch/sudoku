import os
import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract
import matplotlib.pyplot as plt
from skimage.segmentation import clear_border
import imutils

# Charger les modèles YOLO
grid_model = YOLO("./models/best_grid.pt")  # Modèle pour la grande grille
cell_model = YOLO("./models/best_boxes.pt")  # Modèle pour les petites cases

# Fonction de prétraitement de l'image
def extract_digit(cell, debug=False):
    gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray_cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)
    
    if debug:
        cv2.imshow("Cell Thresh", thresh)
        cv2.waitKey(0)
    
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    if len(cnts) == 0:
        return None
    
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)
    
    (h, w) = thresh.shape
    percentFilled = cv2.countNonZero(mask) / float(w * h)
    
    if percentFilled < 0.03:
        return None
    
    digit = cv2.bitwise_and(thresh, thresh, mask=mask)
    
    return digit

# Fonction pour reconnaître un chiffre dans une cellule
def recognize_digit(cell_image, debug=False):
    preprocessed_cell = extract_digit(cell_image, debug)
    
    if preprocessed_cell is not None:
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=123456789'
        result = pytesseract.image_to_string(preprocessed_cell, config=custom_config).strip()
        
        try:
            digit = int(result) if result.isdigit() else 0
        except ValueError:
            digit = 0
    else:
        digit = 0
    
    return digit

# Fonction pour détecter la grande grille et les petites cases
def detect_sudoku_grid(image_path, debug=False):
    img = cv2.imread(image_path)
    
    grid_results = grid_model(image_path, conf=0.5)
    grid_box = None
    for result in grid_results:
        for box in result.boxes:
            box_coords = box.xyxy[0].cpu().numpy().astype(int)
            grid_box = box_coords
    
    if grid_box is not None:
        x1, y1, x2, y2 = grid_box
        img_cropped = img[y1:y2, x1:x2]
        
        if debug:
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))
            plt.title(f"Grille de Sudoku découpée: {image_path}")
            plt.axis("off")
            plt.show()
        
        cell_results = cell_model(img_cropped, conf=0.5)
        boxes = []
        for result in cell_results:
            for box in result.boxes:
                box_coords = box.xyxy[0].cpu().numpy().astype(int)
                boxes.append(box_coords)
        
        height, width = img_cropped.shape[:2]
        cell_height = height // 9
        cell_width = width // 9
        
        grid = np.zeros((9, 9), dtype=np.int32)
        
        for box in boxes:
            x1, y1, x2, y2 = box
            cell = img_cropped[y1:y2, x1:x2]
            
            digit = recognize_digit(cell)
            
            row = int((y1 + y2) / 2 / cell_height)
            col = int((x1 + x2) / 2 / cell_width)
            
            grid[row][col] = digit
        
        return grid
    else:
        return None
