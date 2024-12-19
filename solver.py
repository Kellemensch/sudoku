def is_valid(grid, row, col, num):
    """
    Vérifie si un numéro peut être placé dans la cellule (row, col).
    """
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
    """
    Résout la grille de Sudoku en utilisant l'algorithme de backtracking.
    Renvoie True si la solution est trouvée, sinon False.
    """
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

def solve_grid(grid):
    """
    Résout la grille et renvoie la solution.
    """
    if solve(grid):
        return grid
    else:
        return None
