def extraire_coord(fichier: str) -> tuple[int, dict[int, tuple[float, float]], list[list[float]]]:
    coord = {}
    distances = []
    with open(fichier, 'r') as f:
        # Lire n
        n = int(f.readline().strip())

        # Lire les coordonnées des villes 
        for i in range(n):
            ligne = f.readline().strip()
            x, y = map(float, ligne.split())
            coord[i] = (x, y)
        
        # Lire la matrice de distances
        for _ in range(n):
            ligne = f.readline().strip()
            row = [float(v) for v in ligne.split()]
            distances.append(row)

    return (n, coord, distances)

        

        

print(extraire_coord("instances/instance_10_circle_1.txt")[1])