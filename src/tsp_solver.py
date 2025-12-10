import pulp 
from itertools import combinations 
import time
import sys

def extraire(fichier: str) -> tuple[int, list[list[float]]]:
    """
    Extrait les données importantes d'une instance 
    Args: 
        fichier (str): chemin vers le fichier de l'instance
    Returns:
        n (int): nombre de villes
        distances (list[list[float]]): matrice des distances entre les villes
    """
    distances = []
    with open(fichier, 'r') as f:
        # nombre de villes
        n = int(f.readline().strip())
        # skip les coordonnées
        for i in range(n):
            _ = f.readline()
        # matrice des distances
        for _ in range(n):  
            line = f.readline().strip()
            row = [float(v) for v in line.split()]
            distances.append(row)
    return n, distances

def solve_mtz(distances: list[list[float]], relaxed: bool = False) -> tuple[float, list[int], float, int, int]:
    """
    Résout le TSP en utilisant la formulation MTZ relaxée ou non.
    Args:
        distances (list[list[float]]): matrice des distances entre les villes
        relaxed (bool): si True, utilise la version relaxée
    Returns:
        obj_value (float): valeur de la fonction objective
        tour (list[int]): liste des villes dans l'ordre du tour
        solve_time (float): temps de résolution en secondes
        num_vars (int): nombre de variables dans le modèle
        num_constr (int): nombre de contraintes dans le modèle
    """
    n = len(distances)
    nodes = list(range(n))

    # création du modèle
    prob_name = "MTZ_Relaxed" if relaxed else "MTZ"
    model = pulp.LpProblem(prob_name, pulp.LpMinimize)

    # définition du type de variable
    cat_x = pulp.LpContinuous if relaxed else pulp.LpBinary
    cat_u = pulp.LpContinuous if relaxed else pulp.LpInteger

    # variables de décision
    x = pulp.LpVariable.dicts("x", (nodes, nodes), lowBound=0, upBound=1, cat=cat_x)
    u = pulp.LpVariable.dicts("u", nodes, lowBound=1, upBound=n, cat=cat_u)

    # fonction objectif
    model += pulp.lpSum(distances[i][j] * x[i][j] for i in nodes for j in nodes if i != j)

    # contraintes
    # pas de boucle sur soi-même
    for i in nodes:
        model += x[i][i] == 0

    # une sortie par ville (Contrainte de degré sortant)
    for i in nodes:
        model += pulp.lpSum(x[i][j] for j in nodes if i != j) == 1

    # une entrée par ville (Contrainte de degré entrant)
    for j in nodes:
        model += pulp.lpSum(x[i][j] for i in nodes if i != j) == 1
    
    # élimination des sous-tours (MTZ)
    model += u[0] == 1
    for i in nodes:
        for j in nodes:
            if i != j and i != 0 and j != 0:
                model += u[i] - u[j] + n * x[i][j] <= n - 1

    # time le solveur
    start_time = time.time()
    status = model.solve(pulp.PULP_CBC_CMD(msg=False))
    solve_time = time.time() - start_time

    # vérifie l'optimalité
    if pulp.LpStatus[status] != "Optimal":
        return None, None, solve_time, model.numVariables(), model.numConstraints()

    obj_value = pulp.value(model.objective)
    
    # reconstitue le tour si pas relaxé
    tour = []
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

    return obj_value, tour, solve_time, model.numVariables(), model.numConstraints()

def find_cycles(x_solution, n) -> list[list[int]]:
    """
    Trouve tous les cycles dans la solution donnée avec BFS.
    Args:
        x_solution (dict): dictionnaire des variables de décision x_ij avec leurs valeurs
        n (int): nombre de villes
    Returns:
        cycles (list[list[int]]): liste des cycles trouvés
    """
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

def solve_dfj_it(distances: list[list[float]]) -> tuple[float, list[int], float, int, int, int]:
    """
    Résout le TSP en utilisant la formulation DFJ itérative.
    Args:
        distances (list[list[float]]): matrice des distances entre les villes
    Returns:
        obj_value (float): valeur de la fonction objective
        tour (list[int]): liste des villes dans l'ordre du tour
        total_solve_time (float): temps de résolution total en secondes
        iteration (int): nombre d'itérations effectuées
        num_vars (int): nombre de variables dans le modèle
        num_constr (int): nombre de contraintes dans le modèle final
    """
    n = len(distances)
    nodes = list(range(n))

    model = pulp.LpProblem("DFJ_Iterative", pulp.LpMinimize)

    # variables de décision 
    x = pulp.LpVariable.dicts(
        "x", ((i,j) for i in nodes for j in nodes if i != j),
        cat="Binary"
    )

    # fonction objectif
    model += pulp.lpSum(distances[i][j] * x[(i,j)] for i in nodes for j in nodes if i != j)
    
    # contraintes de degré (entrants et sortants)
    for i in nodes: 
        model += pulp.lpSum(x[(i,j)] for j in nodes if j != i) == 1

    for j in nodes:
        model += pulp.lpSum(x[(i,j)] for i in nodes if i != j) == 1

    # boucle itérative 
    total_solve_time = 0.0
    iteration = 0
    
    while True:
        iteration += 1

        # time le solveur  
        start_t = time.time()
        status = model.solve(pulp.PULP_CBC_CMD(msg=False))
        end_t = time.time()        
        total_solve_time += (end_t - start_t)

        # vérification de l'optimalité
        if pulp.LpStatus[status] != "Optimal":
            print(f"Statut non optimal à l'itération {iteration}")
            return None, None, total_solve_time, iteration, model.numVariables(), model.numConstraints()

        # extraction de la solution courante
        x_solution = {(i, j): pulp.value(x[(i, j)]) for (i, j) in x}
        
        # recherche des cycles avec find_cycles
        cycles = find_cycles(x_solution, n)
        
        # condition d'arrêt : si un seul cycle de longueur n est trouvé 
        if len(cycles) == 1 and len(cycles[0]) == n:
            tour = cycles[0]
            tour.append(tour[0])
            obj_value = pulp.value(model.objective)
            return obj_value, tour, total_solve_time, iteration, model.numVariables(), model.numConstraints()

        # ajout des contraintes d'élimination de sous-tours
        for S in cycles:
            if len(S) < n:
                model += (
                    pulp.lpSum(x[(i, j)] for i in S for j in S if i != j) <= len(S) - 1
                )

def solve_dfj_enum(distances: list[list[float]], relaxed: bool = False) -> tuple[float, list[int], float, int, int]:
    """ 
    Résout le TSP en utilisant la formulation DFJ avec génération a priori des contraintes.
    Args:
        distances (list[list[float]]): matrice des distances entre les villes
        relaxed (bool): si True, utilise une relaxation continue des variables
    Returns:
        obj_value (float): valeur de la fonction objective
        tour (list[int]): liste des villes dans l'ordre du tour
        solve_time (float): temps de résolution en secondes
        num_vars (int): nombre de variables dans le modèle
        num_constr (int): nombre de contraintes dans le modèle
    """
    n = len(distances)
    nodes = list(range(n))

    # création du modèle
    prob_name = "DFJ_Enum_Relaxed" if relaxed else "DFJ_Enum_Integer"
    model = pulp.LpProblem(prob_name, pulp.LpMinimize)

    # type de variable
    cat_x = pulp.LpContinuous if relaxed else pulp.LpBinary

    # variables de décision
    x = pulp.LpVariable.dicts(
        "x",
        ((i, j) for i in nodes for j in nodes if i != j),
        lowBound=0,
        upBound=1,
        cat=cat_x
    )

    # fonction objectif
    model += pulp.lpSum(distances[i][j] * x[(i, j)] for i in nodes for j in nodes if i != j)
    
    # contraintes de degré
    for i in nodes:
        model += pulp.lpSum(x[(i, j)] for j in nodes if j != i) == 1
    for j in nodes:
        model += pulp.lpSum(x[(i, j)] for i in nodes if i != j) == 1

    # contraintes d'élimination de sous-tours 
    for k in range(2, n):
        # pour tout sous-ensemble S de taille k
        for S in combinations(nodes, k):
            model += pulp.lpSum(x[(i, j)] for i in S for j in S if i != j) <= k - 1

    # mesure du temps solveur
    start_time = time.time()
    status = model.solve(pulp.PULP_CBC_CMD(msg=False))
    solve_time = time.time() - start_time

    obj_value = pulp.value(model.objective)

    # reconstitue le tour si pas relaxé et optimal    
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

    return obj_value, tour, solve_time, model.numVariables(), model.numConstraints()

def main():
    filename = sys.argv[1]
    f = int(sys.argv[2])

    _, distances = extraire(f"../instances/{filename}")

    obj, tour, solve_time = None, None, None
    method_name = ""

    if f == 0:
        method_name = "MTZ (Entier)"
        obj, tour, solve_time, _, _  = solve_mtz(distances, False)
    elif f == 1:
        method_name = "MTZ (Relaxé)"
        obj, tour, solve_time, _, _  = solve_mtz(distances, True)
    elif f == 2:
        method_name = "DFJ Enumératif (Entier)"
        obj, tour, solve_time, _, _  = solve_dfj_enum(distances, False)
    elif f == 3:
        method_name = "DFJ Enumératif (Relaxé)"
        obj, tour, solve_time, _, _  = solve_dfj_enum(distances, True)
    elif f == 4:
        method_name = "DFJ Itératif"
        obj, tour, solve_time, iterations, _ , _ = solve_dfj_it(distances)
    

    print(f"Méthode: {method_name}")
    print(f"Valeur objective: {obj}")
    if tour: 
        print(f"Cycle: {tour}")
    elif f in [1,3]:
        print("Cycle: Non disponible pour les versions relaxées.")
    print(f"Temps de résolution: {solve_time:.4f} secondes")
    if f == 4:
        print(f"Nombre d'itérations: {iterations}")

if __name__ == "__main__":
    main()
