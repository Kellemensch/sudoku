# Solver Sudoku à partir d'une image de grille

**Auteurs :  Baptiste Demoisy & Guillaume Nisslé** 

---

## Description

Ce projet est un solveur de Sudoku qui :
1. Identifie une grille de Sudoku dans une image.
2. Reconnaît les chiffres présents dans chaque cellule.
3. Résout la grille et l'affiche sur l'image d'origine.

Le projet est divisé en plusieurs fichiers et dossiers pour une meilleure organisation.

---

## Structure du Projet

### Dossiers Principaux

- **models/** : Contient les modèles YOLO pour détecter la grande grille et les petites cases.
  - `best_grid.pt` : Modèle pour la détection de la grille.
  - `best_boxes.pt` : Modèle pour la détection des cases.

- **test_images/** : Contient les images de test et leurs réponses vérité terrain.
  - `truth_answers.txt` : Fichier contenant les solutions des grilles pour évaluer la précision.


### Fichiers Principaux

- **main.py** : Fichier principal pour exécuter le programme. Permet de choisir entre différents modes (évaluation, résolution, entraînement).
- **test_sudoku.py** : Script pour tester les performances de l'analyseur d'images sur un ensemble d'images de test.
- **analyzer.py** : Contient toutes les fonctions d'analyse d'image, de grilles, de cases et de préprocessing.
- **solver.py** : Contient la logique pour résoudre une grille de Sudoku.
- **afficheur.py** : Annoter les solutions sur l'image d'origine et sauvegarder le résultat.
- **train_model.py** : Script pour entraîner ou affiner les modèles YOLO.

---

## Utilisation

Le programme peut être utilisé en ligne de commande avec plusieurs arguments.

### Arguments Disponibles

1. **Mode résolution d'une image**
   ```bash
   python3 main.py --image <path_to_image> --debug
   ```
   - `--image_path` : Chemin vers l'image contenant la grille de Sudoku.
   - `--debug` : Permet l'affichage des grilles après analyse (Optionnel)

2. **Mode évaluation des performances**
   ```bash
   python3 main.py --test --debug
   ```
   - `--test` : Lance les tests sur toutes les images du dossier ./test_images/.
   - `--debug` : Permet l'affichage des grilles après analyse (Optionnel)

3. **Mode entraînement des modèles**
   ```bash
   python3 main.py --train
   ```
   - `--train` : lance l'entrainement des modèles de reconnaissance de grille.

---

## Fonctionnement Global

1. **Détection de la grille** :
   - `main.py` appelle les fonctions de détection dans `analyzer.py`.
   - Les modèles YOLO sont utilisés pour détecter la grille et les cases.

2. **Reconnaissance des chiffres** :
   - `analyzer.py` utilise Tesseract pour reconnaître les chiffres présents dans chaque cellule.

3. **Résolution de la grille** :
   - `solver.py` contient la logique pour résoudre une grille de Sudoku.

4. **Affichage des solutions** :
   - `afficheur.py` ajoute les chiffres de la solution sur l'image originale.

5. **Entraînement des modèles** :
   - `train.py` est utilisé pour (ré)entraîner les modèles YOLO avec de nouvelles données.

---

## Exemple d’Exécution

### Résoudre une image unique

```bash
python3 main.py --image ./test_images/test.jpg
```

### Tester sur plusieurs images

```bash
python3 main.py test --test
```

### Entraîner les modèles

```bash
python3 main.py --train 
```

---

## Prérequis

Un fichier est disponible afin de télécharger toutes les dépendances pip :

```bash
pip install -r requirements.txt
```

- **Python 3.8+**
- Bibliothèques Python :
  - `opencv-python`
  - `numpy`
  - `pytesseract`
  - `ultralytics`
  - `matplotlib`
  - `scikit-image`
  - `imutils`

- **Tesseract OCR** 

---

## Notes

- Assurez-vous que les modèles YOLO sont correctement entraînés et disponibles dans le dossier `models/`.
- Pour des tests personnalisés, modifiez ou ajoutez des images dans le dossier `test_images/` et mettez à jour le fichier `truth_answers.txt`.

---

