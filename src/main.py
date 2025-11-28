import pulp 
from itertools import combinations 

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


def find_cycles(x_solution, n):
    visited = [False] * n
    cycles = []

    for start in range(n):
        if not visited[start]:
            current = start
            cycle = []

            while not visited[current]:
                visited[current] = True
                cycle.append(current)

                next_nodes = [j for (i,j) in x_solution if i == current and x_solution[(i,j)] > 0.5]
                if not next_nodes:
                    break
                current = next_nodes[0]
            
            if len(cycle) >= 2:
                cycles.append(cycle)
    
    return cycles

def dfj_model(n, distances):
    V = range(n)
    model = pulp.LpProblem("DFJ", pulp.LpMinimize)

    x = pulp.LpVariable.dicts(
        "x", ((i,j) for i in V for j in V if i != j),
        cat="Binary"
    )

    model += pulp.lpSum(distances[i][j] * x[(i,j)] for i in V for j in V if i != j)

    for i in V: 
        model += pulp.lpSum(x[(i,j)] for j in V if j != i) == 1

    for j in V:
        model += pulp.lpSum(x[(i,j)] for i in V if i != j) == 1

    return model, x


def solve_dfj_it(distances):
    n = len(distances)
    model, x = dfj_model(n, distances)

    iteration = 0

    while True:
        model.solve(pulp.PULP_CBC_CMD(msg=False))

        x_solution = {(i,j): pulp.value(x[(i,j)])
                    for (i, j) in x}
        
        cycles = find_cycles(x_solution, n)

        if len(cycles) == 1 and len(cycles[0]) == n:
            print("sol optimale trouvée.")
            break

        for S in cycles:
            if len(S) >= 2 and len(S) < n:
                model += (
                    pulp.lpSum(x[(i,j)]
                               for i in S for j in S if i != j)
                    <= len(S) - 1
                )
        
        iteration += 1

    return cycles[0], pulp.value(model.objective)

def solve_dfj_enum(distances):
    n = len(distances)
    V = range(n)

    model = pulp.LpProblem("DFJ_ENUM", pulp.LpMinimize)

    x = pulp.LpVariable.dicts(
        "x",
        ((i,j) for i in V for j in V if i != j),
        lowBound=0,
        upBound=1,
        cat="Binary"
    )

    model += pulp.lpSum(distances[i][j] * x[(i, j)]
                        for i in V for j in V if i != j)
    
    for i in V:
        model += pulp.lpSum(x[(i, j)] for j in V if j != i) == 1

    for j in V:
        model += pulp.lpSum(x[(i, j)] for i in V if i != j) == 1

    for k in range(2, n):
        for S in combinations(V, k):
            S = list(S)
            model += (
                pulp.lpSum(
                    x[(i, j)] for i in S for j in S if i != j
                ) <= k - 1
            )

    cost = model.solve(pulp.PULP_CBC_CMD(msg=False))
    cost = pulp.value(model.objective)

    return cost, x

if __name__ == "__main__":
    fichier = "../instances/instance_10_euclidean_1.txt"
    distances = extraire(fichier)[2]
    cost, tour, arcs = solve_mtz(distances)


    cycles, value = solve_dfj_it(distances)
    print(cycles, value)

    print("+++++")
    
    costB, x = solve_dfj_enum(distances)
    print(costB)

    print("+++++")
     
    if tour is None:
        print("Statut non optimal.")
    else:
        print("Coût optimal:", cost)
        print("Tour:", tour)
        print("Arcs:", arcs)  