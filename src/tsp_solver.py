from solvers import extraire, solve_mtz, solve_dfj_it, solve_dfj_enum
import sys



def main():
    filename = sys.argv[1]
    f = int(sys.argv[2])

    distances = extraire(f"../instances/{filename}")

    obj, tour, solve_time = None, None, None
    method_name = ""

    if f == 0:
        method_name = "MTZ (Entier)"
        obj, tour, solve_time, n, _ , _ = solve_mtz(distances, False)
    elif f == 1:
        method_name = "MTZ (Relaxé)"
        obj, tour, solve_time, n, _ , _ = solve_mtz(distances, True)
    elif f == 2:
        method_name = "DFJ Enumératif (Entier)"
        obj, tour, solve_time, n, _ , _ = solve_dfj_enum(distances, False)
    elif f == 3:
        method_name = "DFJ Enumératif (Relaxé)"
        obj, tour, solve_time, n, _ , _ = solve_dfj_enum(distances, True)
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





