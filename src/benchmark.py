import os
import csv
import re

from solvers import extraire, solve_mtz, solve_dfj_it, solve_dfj_enum

INSTANCES_DIR = "../instances"
OUTPUT_CSV = "results.csv"

def benchmark():
    with open(OUTPUT_CSV, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ["instance", "formulation", "obj_int", "time_int", "obj_relax", "time_relax", "gap", "vars", "constr"]
        writer.writerow(header)

        files = [f for f in os.listdir(INSTANCES_DIR)]

        for filename in files:
            filepath =  os.path.join(INSTANCES_DIR, filename)


            n, distances = extraire(filepath)
            instance_name = filename.replace('.txt', '')
            # MTZ 
            print(f"  > MTZ Integer...")
            obj_int, _, time_int, vars_int, constr_int = solve_mtz(distances, relaxed=False)
            print(f"  > MTZ Relax...")
            obj_relax, _, time_relax, _, _ = solve_mtz(distances, relaxed=True)
            gap = (obj_int - obj_relax) / obj_int if obj_int else 0.0
            writer.writerow([instance_name, "MTZ", round(obj_int, 2), round(time_int, 4), 
                             round(obj_relax, 2), round(time_relax, 4), round(gap, 4), 
                             vars_int, constr_int])
            
            # DFJ Itératif
            print(f"  > DFJ Iterative...")
            obj_iter, _, time_iter, iter_count, vars_iter, constr_iter = solve_dfj_it(distances)
            writer.writerow([instance_name, "DFJ_iter", round(obj_iter, 2), round(time_iter, 4), 
                             "-", "-", "-", vars_iter, constr_iter])
            
            # DFJ Enumératif
            if n < 20:
                print(f"  > DFJ Enum Integer...")
                obj_enum, _, time_enum, vars_enum, constr_enum = solve_dfj_enum(distances, relaxed=False)
                print(f"  > DFJ Enum Relax...")
                obj_relax_enum, _, time_relax_enum, _, _ = solve_dfj_enum(distances, relaxed=True)
                gap_enum = (obj_enum - obj_relax_enum) / obj_enum if obj_enum else 0.0
                writer.writerow([instance_name, "DFJ_enum", round(obj_enum, 2), round(time_enum, 4), 
                                 round(obj_relax_enum, 2), round(time_relax_enum, 4), round(gap_enum, 4), 
                                 vars_enum, constr_enum])
            else:
                print(f"  > DFJ Enum ignoré (n={n} >= 20)")

            f.flush()
        print(f"\nTerminé ! Résultats sauvegardés dans {OUTPUT_CSV}")

if __name__ == "__main__":
    benchmark()