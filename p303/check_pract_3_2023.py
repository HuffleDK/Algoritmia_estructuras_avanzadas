#!/usr/bin/env python
# coding: utf-8
import argparse
import textwrap
import sys

import numpy as np

from typing import List, Tuple, Dict, Callable, Union

def main(size=10, shift=3):
    """
    """
    ##### quick select
    
    print(30*'_' + "Checking qsel")

    t = np.random.permutation(size)
    for k in range(0, len(t), 2):
        print('  qs', k + shift, p3.qsel(t, k+shift))
        print('qsnr', k + shift, p3.qsel_nr(t, k+shift))

    _ = input("\npulsar Intro para continuar " + 20*'.' + "\n")
    
    print(30*'_' + "Checking qsel5")
    
    t = np.random.permutation(size).astype(int)
    for k in range(len(t)):
        print(k + shift, p3.qsel5_nr(t, k+shift))

    _ = input("\npulsar Intro para continuar " + 20*'.' + "\n")
    
    print(30*'_' + "Checking qsort 5")

    for _ in range(10):
        t = np.random.permutation(size).astype(int)
        print(p3.qsort_5(t))
    
    _ = input("\npulsar Intro para continuar " + 20*'.' + "\n")
    
    ##### Dynamic Programming for change
    print(30*'_' + "change PD")
    
    l_coins = [1, 2, 5, 10] #, 20, 50, 100, 200, 500]
    l_coins = [1, 3, 5, 7]
    
    print('coins:', l_coins)
    for c in range(15, 26):
        val = p3.change_pd(c, l_coins)[-1, -1]
        d_ch = p3.optimal_change_pd(c, l_coins)
        print('change pd: c', c, 'val', val, 'comp', list(d_ch.items()))
    
    _ = input("\npulsar Intro para continuar " + 20*'.' + "\n")
    
    ##### greedy knapsack
    print(30*'_' + "knapsack greedy")
    
    l_weights = [4, 4, 5]
    l_values = [10, 11, 15]
    
    print('weights:', l_weights)
    print(' values:', l_values)
    
    for bound in range(4, 14):
        d_ch = p3.knapsack_fract_greedy(l_weights, l_values, bound)
        print('   gr knapsack: b', bound, '\tval', (np.array(list(d_ch.values())) * np.array(l_values) / np.array(l_weights)).sum(), '\tcomp', list(d_ch.items()))
    
    _ = input("\npulsar Intro para continuar " + 20*'.' + "\n")
    
    ##### Dynamic Programming for knapsack
    print(30*'_' + "Checking mochila 0-1")

    for bound in range(4, 14):
        val = p3.knapsack_01_pd(l_weights, l_values, bound=bound)
        print('01 knapsack pd: b', bound, 'val', val)
    

###############################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
        """
        Comprobador de la práctica 3. 
        
        Comprueba: 
            * quickselect y quickselect5
            * quicksort
            * change pd
            * mochila 0-1 pd
            
        Args: 
            size
            shift
            
        Ejecutar por ejemplo en Linux como 
        
            ./check_pract_3_2023.py -si 20 
            
        y en Windows como 
        
            python check_pract_3_2023.py -sh -5
        """))
    
    #parser.add_argument("-p", "--pareja", type=str, default=None)
    parser.add_argument("-sz", "--size", help="tamaño de permutaciones; default=20", type=int, default=20)
    parser.add_argument("-sh", "--shift", help="shift de claves (positivo o negativo); default=0, no shift", type=int, default=0)
    
    args = parser.parse_args()
    
    if len(sys.argv) > 1:
        import p303 as p3 
            
        main(args.size, args.shift)
    
    else:
        parser.print_help()
            
        