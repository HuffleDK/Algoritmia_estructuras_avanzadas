#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import argparse
import textwrap

import numpy as np

# cambiar p100 por p1NN
import p103 as p1


####################################### main
def main(t_size: int):
    """Comprobador de practica 1.

    Comprueba BB, min heap inserción t eliminación, PQ creación y extracción y
    heap based selection.
    BB, PQ y selection se prueban en tablas aleatorias; heaps sobre tablas fijas

    Args:
        t_size: tamaño de tablas para bb y selección

    Returns:
        None
    """
    # check bb recursiva e iterativa
    print(30 * "_" + "Checking recursive and iterative BB")

    print(
        "\t se genera un array ordenado aleatoria y se buscan sus sucesivos elementos"
        + "\n\t el bucle de busqueda solo se interrumpe en caso de error."
        + "\n\n\t La ejecucion ha sido correcta si no hay interrupciones"
    )
    t = np.random.permutation(2 * t_size)[:t_size]
    t = np.sort(t[:t_size])

    # el bucle se interrumpe en caso de error
    for k in range(len(t)):
        idx_rec = p1.rec_bb(t, 0, len(t) - 1, t[k])
        if idx_rec != k:
            print("error en bb recursiva")
            break

        idx_iter = p1.bb(t, 0, len(t) - 1, t[k])
        if idx_iter != k:
            print("error en bb no recursiva")
            break

    _ = input("\npulsar Intro para continuar " + 20 * "." + "\n")

    # check min heap;
    print(30 * "_" + "Checking min heap insertion")

    # se insertan sucesivamente en un min heap los elementos de un array ordenado cuyo min heap
    # correspondiente se conoce y se imprime el min heap final para comprobar si
    # la creación ha sido correcta

    # cambiar por cualquier array cuyo min heap se conoozca
    t = np.random.permutation(2 * t_size)[:t_size]
    t = np.sort(t[:t_size])

    print(t)
    h = np.array([], dtype=int)
    for i in range(len(t)):
        h = p1.insert_min_heap(h, t[i])
    print(h)
    print("ejecucion correcta si los dos prints coinciden\n")

    _ = input("\npulsar Intro para continuar " + 20 * "." + "\n")

    print(30 * "_" + "Checking min heap creation")

    # mismo objetivo que la prueba anterior pero ahora el min heap se genera directamente sobre el array dado
    # el min heap final puede ser distinto del anterior
    for _ in range(3):
        t = np.random.permutation(2 * t_size)[:t_size]
        p1.create_min_heap(t)
        print(t)
    print("comprobar visualmente que se han obtenido min heaps\n")

    # check pq
    _ = input("\npulsar Intro para continuar " + 20 * "." + "\n")

    print(30 * "_" + "Checking PQ insertion and removal")

    # se genera un array aleatorio, se insertan sus elementos en una pq y se extraen a continuación
    # lo que debe producir una ordenacion del array

    # t = np.random.permutation(2 * t_size)[ : t_size]

    pq_ = p1.pq_ini()
    for i in range(len(t)):
        pq_ = p1.pq_insert(pq_, t[i])

    for _ in range(len(t)):
        e, pq_ = p1.pq_remove(pq_)
        print(e)
    print("ejecucion correcta si los elementos se imprimen de manera ordenada\n")

    # check sorting on min heaps
    _ = input("\npulsar Intro para continuar " + 20 * "." + "\n")

    print(30 * "_" + "Checking min heap based sorting")

    # se genera arrays aleatorios y se ordenan mediante min heaps

    for _ in range(3):
        t = np.random.permutation(t_size)
        t = np.random.permutation(2 * t_size)[:t_size]
        # t = np.random.permutation(2 * t_size)[ : t_size]
        t2 = np.copy(t)

        print("\n    ", np.sort(t))
        print("mhs:", p1.min_heap_sort(t2))
    print("ejecucion correcta si los dos arrays coinciden\n")


###############################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
        Comprobador de la práctica 1. 
        
        Comprueba BB, min heap inserción t eliminación, PQ creación y extracción y 
        min heap based sorting. 
        
        Ejecutar por ejemplo en Linux como 
        
            ./check_pract_1_2022.py -s 10 
            
        y en Windows como 
        
            python check_pract_1_2022.py -s 10
        """
        ),
    )

    parser.add_argument("-p", "--pareja", type=str, default="00")
    parser.add_argument("-s", "--size", help="size of arrays", type=int, default=10)

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()

    else:
        # import p100 as p1

        main(args.size)
