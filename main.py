import torch
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

# Charger les modèles YOLO
grid_model = YOLO("./models/best_grid.pt")  # Modèle pour la grande grille
cell_model = YOLO("./models/best_boxes.pt")  # Modèle pour les petites cases
# digits_model = YOLO("./models/best_digits.pt") # Modèle pour les chiffres

# Charger l'image
image_path = "./test2.jpg"
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Détection de la grande grille
results = grid_model(image_path, conf=0.5)

# Trouver la boîte englobante de la grille
for result in results:
    boxes = result.boxes.cpu().numpy()
    for box in boxes:
        r = box.xyxy[0].astype(int)  # Coins de la grande grille
        class_id = box.cls[0].astype(int)
        confidences = box.conf[0].astype(float)

        # Dessiner la grande grille
        cv2.rectangle(img_rgb, r[:2], r[2:], (0, 255, 0), 2)
        break  # On ne prend qu'une seule grille

# Découper l'image pour obtenir la grille
cropped = img[r[1]:r[3], r[0]:r[2]]
cropped_rgb = img_rgb[r[1]:r[3], r[0]:r[2]]
cv2.imwrite("result.jpg",img_rgb)

# Générer une grille fixe 9x9 basée sur la grande grille
grid_height, grid_width = cropped.shape[:2]
cell_height = grid_height // 9
cell_width = grid_width // 9

# Initialiser la grille Sudoku
sudoku_grid = np.zeros((9, 9), dtype=int)

def preprocess_cell(cell):
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    _, binary = cv2.threshold(resized, 128, 255, cv2.THRESH_BINARY_INV)
    return binary


# Détection des petites cases
for i in range(9):
    for j in range(9):
        # Découper chaque petite case
        x1, y1 = j * cell_width, i * cell_height
        x2, y2 = (j + 1) * cell_width, (i + 1) * cell_height
        cell = cropped[y1:y2, x1:x2]

        # Agrandir chaque petite case pour améliorer la reconnaissance des chiffres
        cell_resized = cv2.resize(cell, (28, 28), interpolation=cv2.INTER_LINEAR)

        # Prétraitement : Convertir en niveaux de gris et appliquer un seuillage
        gray_cell = cv2.cvtColor(cell_resized, cv2.COLOR_BGR2GRAY)
        _, binary_cell = cv2.threshold(gray_cell, 128, 255, cv2.THRESH_BINARY_INV)

        # Nettoyage des petites lignes parasites
        kernel = np.ones((2, 2), np.uint8)
        cleaned_cell = cv2.morphologyEx(binary_cell, cv2.MORPH_CLOSE, kernel)

        # Reconnaissance des chiffres avec Tesseract
        digit = pytesseract.image_to_string(
            cleaned_cell,
            config="--psm 10 -c tessedit_char_whitelist=0123456789"
            ).strip()

        if digit.isdigit():  # Vérifier si un chiffre a été reconnu
            sudoku_grid[i, j] = int(digit)

        # Dessiner la case et le chiffre détecté
        cv2.rectangle(cropped_rgb, (x1, y1), (x2, y2), (255, 0, 0), 1)
        if digit.isdigit():
            cv2.putText(
                cropped_rgb,
                digit,
                (x1 + 10, y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

# Afficher et sauvegarder les résultats
cv2.imwrite("result_cells.jpg", cropped_rgb)
plt.imshow(cropped_rgb)
plt.title("Petites cases détectées avec chiffres")
plt.show()

print("Grille Sudoku détectée :")
print(sudoku_grid)




def solve_sudoku(board):
    """
    Résout une grille Sudoku (algorithme de backtracking).
    """
    def is_valid(board, row, col, num):
        # Vérifier si 'num' peut être placé dans la case (row, col)
        for i in range(9):
            if board[row][i] == num or board[i][col] == num:
                return False
            if board[row//3*3 + i//3][col//3*3 + i%3] == num:
                return False
        return True

    def solve():
        for row in range(9):
            for col in range(9):
                if board[row][col] == 0:  # Case vide
                    for num in range(1, 10):  # Essayer tous les chiffres
                        if is_valid(board, row, col, num):
                            board[row][col] = num
                            if solve():
                                return True
                            board[row][col] = 0  # Backtrack
                    return False
        return True

    solve()
    return board

# Résoudre la grille détectée
solved_grid = solve_sudoku(sudoku_grid)
print("Grille Sudoku résolue :")
print(np.array(solved_grid))
