import pulp 
from itertools import combinations 
import time

def extraire(fichier: str) -> tuple[int, dict[int, tuple[float, float]], list[list[float]]]:
    distances = []
    with open(fichier, 'r') as f:
        # Lire n
        n = int(f.readline().strip())

        # Skip les coordonnées des villes 
        for i in range(n):
            ligne = f.readline()
        
        # Lire la matrice de distances
        for _ in range(n):
            ligne = f.readline().strip()
            row = [float(v) for v in ligne.split()]
            distances.append(row)

    return distances

def solve_mtz(distances: list[list[float]], relaxed: bool = False):
    n = len(distances)
    nodes = list(range(n))

    prob_name = "MTZ_Relaxed" if relaxed else "MTZ"
    model = pulp.LpProblem(prob_name, pulp.LpMinimize)

    cat_x = pulp.LpContinuous if relaxed else pulp.LpBinary
    cat_u = pulp.LpContinuous if relaxed else pulp.LpInteger

    # Variables de décision
    # x_ij : 1 si l'arc (i,j) est utilisé, 0 sinon (ou réel entre 0 et 1 si relaxé)
    x = pulp.LpVariable.dicts("x", (nodes, nodes), lowBound=0, upBound=1, cat=cat_x)
    # u_i : variable auxiliaire pour l'ordre de visite
    u = pulp.LpVariable.dicts("u", nodes, lowBound=1, upBound=n, cat=cat_u)

    # Fonction objectif
    model += pulp.lpSum(distances[i][j] * x[i][j] for i in nodes for j in nodes if i != j)

    # Contraintes
    # Pas de boucle sur soi-même
    for i in nodes:
        model += x[i][i] == 0

    # Une sortie par ville (Contrainte de degré sortant)
    for i in nodes:
        model += pulp.lpSum(x[i][j] for j in nodes if i != j) == 1

    # Une entrée par ville (Contrainte de degré entrant)
    for j in nodes:
        model += pulp.lpSum(x[i][j] for i in nodes if i != j) == 1
    
    # Élimination des sous-tours (MTZ)
    model += u[0] == 1

    for i in nodes:
        for j in nodes:
            if i != j and i != 0 and j != 0:
                model += u[i] - u[j] + n * x[i][j] <= n - 1

    start_time = time.time()
    status = model.solve(pulp.PULP_CBC_CMD(msg=False))
    solve_time = time.time() - start_time

    if pulp.LpStatus[status] != "Optimal":
        return None, None, solve_time

    obj_value = pulp.value(model.objective)
    
    tour = []
    # Reconstruction du tour à partir des variables x si non relaxé car sinon les valeurs sont fractionnaires
    if not relaxed:
        selected = [(i, j) for i in nodes for j in nodes if pulp.value(x[i][j]) > 0.5]
        succ = {i: j for i, j in selected}
        curr = 0
        visited = set()
        while len(visited) < n:
            tour.append(curr)
            visited.add(curr)
            if curr in succ:
                curr = succ[curr]
            else:
                break
        tour.append(0) 

    return obj_value, tour, solve_time

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
    nodes = list(range(n))
    model = pulp.LpProblem("DFJ", pulp.LpMinimize)

    x = pulp.LpVariable.dicts(
        "x", ((i,j) for i in nodes for j in nodes if i != j),
        cat="Binary"
    )

    model += pulp.lpSum(distances[i][j] * x[(i,j)] for i in nodes for j in nodes if i != j)
    for i in nodes: 
        model += pulp.lpSum(x[(i,j)] for j in nodes if j != i) == 1

    for j in nodes:
        model += pulp.lpSum(x[(i,j)] for i in nodes if i != j) == 1

    return model, x

def solve_dfj_it(distances: list[list[float]]):
    n = len(distances)
    
    model, x = dfj_model(n, distances)
    
    total_solve_time = 0.0
    iteration = 0
    
    while True:
        iteration += 1

        start_t = time.time()
        status = model.solve(pulp.PULP_CBC_CMD(msg=False))
        end_t = time.time()        
        total_solve_time += (end_t - start_t)

        if pulp.LpStatus[status] != "Optimal":
            print(f"Statut non optimal à l'itération {iteration}")
            return None, None, total_solve_time, iteration

        x_solution = {(i, j): pulp.value(x[(i, j)]) for (i, j) in x}
        
        cycles = find_cycles(x_solution, n)
        
        if len(cycles) == 1 and len(cycles[0]) == n:
            tour = cycles[0]
            tour.append(tour[0])
            obj_value = pulp.value(model.objective)
            return obj_value, tour, total_solve_time, iteration

        for S in cycles:
            if len(S) < n:
                model += (
                    pulp.lpSum(x[(i, j)] for i in S for j in S if i != j) <= len(S) - 1
                )

def solve_dfj_enum(distances: list[list[float]], relaxed: bool = False):
    n = len(distances)
    nodes = list(range(n))

    prob_name = "DFJ_Enum_Relaxed" if relaxed else "DFJ_Enum_Integer"
    model = pulp.LpProblem(prob_name, pulp.LpMinimize)

    # Définition du type de variable
    cat_x = pulp.LpContinuous if relaxed else pulp.LpBinary

    # Variables x_ij
    x = pulp.LpVariable.dicts(
        "x",
        ((i, j) for i in nodes for j in nodes if i != j),
        lowBound=0,
        upBound=1,
        cat=cat_x
    )

    # Fonction objectif
    model += pulp.lpSum(distances[i][j] * x[(i, j)] for i in nodes for j in nodes if i != j)
    
    # Contraintes de degré
    for i in nodes:
        model += pulp.lpSum(x[(i, j)] for j in nodes if j != i) == 1
    for j in nodes:
        model += pulp.lpSum(x[(i, j)] for i in nodes if i != j) == 1

    # Contraintes d'élimination de sous-tours (Génération a priori)
    for k in range(2, n):
        # Pour tout sous-ensemble S de taille k
        for S in combinations(nodes, k):
            model += pulp.lpSum(x[(i, j)] for i in S for j in S if i != j) <= k - 1

    # Mesure du temps solveur
    start_time = time.time()
    status = model.solve(pulp.PULP_CBC_CMD(msg=False))
    solve_time = time.time() - start_time

    obj_value = pulp.value(model.objective)
    
    tour = []
    if not relaxed and pulp.LpStatus[status] == "Optimal":
        selected = [(i, j) for (i, j) in x if pulp.value(x[(i, j)]) > 0.5]
        succ = {i: j for i, j in selected}
        curr = 0
        visited = set()
        while len(visited) < n:
            tour.append(curr)
            visited.add(curr)
            if curr in succ:
                curr = succ[curr]
            else:
                break
        tour.append(0)

    return obj_value, tour, solve_time

