#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import argparse
import textwrap

import numpy as np
from typing import Dict


def cd_2_dict(p: np.ndarray) -> Dict:
    """Transforms a DS over an array p into a dict with the representatives
    as the dict keys and the subsets as list in the dict values
    """
    d_cd = {}

    # representatives as keys in the dict
    for u in range(len(p)):
        if p[u] < 0:
            d_cd[u] = []

    # subsets as list as dict values
    for u in range(len(p)):
        r_u = p2.find(u, p)
        d_cd[r_u].append(u)

    return d_cd


####################################### main
def main():
    """Comprobador de practica 2.

    Comprueba gestión de CDs, identificación de componentes conexas.

    Args:
        t_size: tamaño de tablas para bb y selección

    Returns:
        None
    """
    # check cd
    print(30 * "_" + "Checking CD")

    # ejemplo del problema 2.15; usar cualquier otro
    p = np.array([-3, 0, 1, -3, 3, 4, -1, -3, 7, 8])
    print("tabla cd antes\t", p)

    d = cd_2_dict(p.copy())
    print("antes  \t", d)

    p2.union(p2.find(1, p), p2.find(5, p), p)
    p2.union(p2.find(2, p), p2.find(6, p), p)
    p2.union(p2.find(5, p), p2.find(9, p), p)
    print("tabla cd después\t", p)

    d = cd_2_dict(p)
    print(" dict cd después\t", d)

    _ = input("\npulsar Intro para continuar " + 20 * "." + "\n")

    # check cc
    print(30 * "_" + "Checking Kruskal")

    # ejemplo de grafo; usar cualquier otro
    n0 = 4
    l_g0 = [(0, 1, 1), (0, 2, 2), (1, 2, 3), (1, 3, 4), (2, 3, 5)]

    print(p2.kruskal(n0, l_g0))

    _ = input("\npulsar Intro para continuar " + 20 * "." + "\n")

    # check tsp
    print(30 * "_" + "Checking greedy TSP")
    # distances for ["barcelona", "madrid", "sevilla", "valencia"]
    dist_bmsv = np.array(
        [[0, 624, 995, 350], [624, 0, 506, 357], [995, 506, 0, 653], [350, 357, 653, 0]]
    )

    # simple graph
    dist_simple = np.array([[0, 1, 2, 1], [1, 0, 1, 2], [2, 1, 0, 1], [1, 2, 1, 0]])

    # coger uno de los dos anteriores
    dist_m = dist_simple
    print("graph_matrix\n", dist_m)

    circuit = p2.greedy_tsp(dist_m)
    print(circuit)

    circuit_length = p2.len_circuit(circuit, dist_m)
    print("\ngreedy length: %d" % circuit_length)

    _ = input("\npulsar Intro para continuar " + 20 * "." + "\n")

    # check repeated greedy tsp
    print(30 * "_" + "Checking iterated greedy TSP")

    circuit = p2.repeated_greedy_tsp(dist_m)
    print(circuit)

    circuit_length = p2.len_circuit(circuit, dist_m)
    print("\nrepeated greedy length: %d" % circuit_length)

    _ = input("\npulsar Intro para continuar " + 20 * "." + "\n")

    # check exhaustive tsp
    print(30 * "_" + "Checking exhaustive TSP")

    circuit = p2.exhaustive_tsp(dist_m)
    print(circuit)

    circuit_length = p2.len_circuit(circuit, dist_m)
    print("\nexhaustive length: %d" % circuit_length)

    _ = input("\npulsar Intro para continuar " + 20 * "." + "\n")

    # check on random graphs
    print(30 * "_" + "Checking on random graphs")

    for _ in range(5):
        _ = input("\npulsar Intro para continuar " + 20 * "." + "\n")

        dm = p2.dist_matrix(7, w_max=50)

        c_g = p2.greedy_tsp(dm)
        l_g = p2.len_circuit(c_g, dm)

        c_rg = p2.repeated_greedy_tsp(dm)
        l_rg = p2.len_circuit(c_rg, dm)

        c_e = p2.exhaustive_tsp(dm)
        l_e = p2.len_circuit(c_e, dm)

        # if l_e < l_g:
        print("\n", dm)
        print("\n         greedy length: %d km" % l_g)
        print("greedy circuit", c_g)

        print("\nrepeated greedy length: %d km" % l_rg)
        print("repeated greedy circuit", c_rg)

        print("\n     exhaustive length: %d km" % l_e)
        print("exhaustive circuit", c_e)


###############################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
        Comprobador de la práctica 1. 
        
        Comprueba BB, min heap inserción t eliminación, PQ creación y extracción y 
        heap based selection. 
        
        BB, PQ y selection se prueban en tablas aleatorias; heaps sobre tablas fijas.
        
        Ejecutar por ejemplo en Linux como 
        
            ./check_pract_1_2022.py -s 10 
            
        y en Windows como 
        
            python check_pract_1_2022.py -s 10
        """
        ),
    )

    parser.add_argument("-p", "--pareja", type=str, default=None)

    args = parser.parse_args()

    if len(sys.argv) > 1:
        if args.pareja is not None:
            f_path = "./p2" + args.pareja + ".py"
            if os.path.isfile(f_path):
                str_comm = "cp " + f_path + " p2.py"
                os.system(str_comm)
                import p203

                main()

            else:
                sys.exit("file {0:s} not found".format(f_path))

    else:
        parser.print_help()
