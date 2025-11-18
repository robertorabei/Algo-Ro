import pulp 

def extraire(fichier: str) -> tuple[int, dict[int, tuple[float, float]], list[list[float]]]:
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


def solve_mtz(distances : list[list[float]]):
    n = len(distances)
    noeuds = list(range(n))

    model = pulp.LpProblem("MTZ", pulp.LpMinimize)

    # Variables de décision
    x = pulp.LpVariable.dicts("x", (noeuds, noeuds), lowBound = 0, upBound = 1, cat="Binary")
    u = pulp.LpVariable.dicts("u", noeuds, lowBound = 1, upBound = n, cat="Integer")

    model += pulp.lpSum(distances[i][j] * x[i][j] for i in noeuds for j in noeuds)

    # Pas de noeuds
    for i in noeuds:
        model += x[i][i] == 0

    # Une sortie par ville
    for i in noeuds:
        model += pulp.lpSum(x[i][j] for j in noeuds) == 1

    # Une entrée par ville
    for j in noeuds:
        model += pulp.lpSum(x[i][j] for i in noeuds) == 1

    # Eliminer sous-tours
    # Fixer u_0 = 1
    model += u[0] == 1
    # Pour i >= 1, u_i >= 2
    for i in noeuds[1:]:
        model += u[i] >= 2

    # Contraintes MTZ
    for i in noeuds:
        for j in noeuds:
            if i != j:
                if i != 0 and j != 0:
                    model += u[i] - u[j] + n*x[i][j] <= n-1
    
    # Résolution
    status = model.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # Vérification du statut
    if pulp.LpStatus[status] != "Optimal":
        return None, None, None
                
    # Extraire les arcs sélectionnés
    selected = [(i, j) for i in noeuds for j in noeuds if pulp.value(x[i][j]) > 0.5]
    
    # Reconstruire le tour à partir des arcs
    succ = {i: j for i, j in selected}
    tour = [0]
    while len(tour) < n:
        tour.append(succ[tour[-1]])
    # fermer le cycle
    tour.append(0)

    cost = sum(distances[tour[k]][tour[k+1]] for k in range(len(tour)-1))
    return cost, tour, selected

if __name__ == "__main__":
    fichier = "../instances/instance_10_circle_1.txt"
    distances = extraire(fichier)[2]
    cost, tour, arcs = solve_mtz(distances)

    if tour is None:
        print("Statut non optimal.")
    else:
        print("Coût optimal:", cost)
        print("Tour:", tour)
        print("Arcs:", arcs)  