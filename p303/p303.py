from typing import Dict, List, Tuple, Union
from collections import OrderedDict

import numpy as np

def split(t: np.ndarray)-> Tuple[np.ndarray, int, np.ndarray]:
    """Reparte los elementos de t en dos listas con menores y mayores que t[0]

    Args:
        t (np.ndarray): Lista de elementos

    Returns:
        Tuple[np.ndarray, int, np.ndarray]: Lista de menores, pivot y lista de mayores, en ese orden
    """
    menores = np.array([elemento for elemento in t if elemento < t[0]])
    mayores = np.array([elemento for elemento in t if elemento > t[0]])
    return (menores, t[0], mayores)

def qsel(t: np.ndarray, k: int)-> Union[int, None]:
    """Selecciona el elemento en el indice k en un t ordenado

    Args:
        t (np.ndarray): lista de elementos
        k (int): indice a buscar

    Returns:
        Union[int, None]: Indice en la posicion k o None
    """
    if len(t) == 1 and k == 0:
        return t[0]
    if len(t) == 0:
        return None
    
    t_l, mid, t_r = split(t)
    m = len(t_l)
    if k == m:
        return mid
    elif k < m:
        return qsel(t_l, k)
    else:
        return qsel(t_r, k-m-1)

def qsel_nr(t: np.ndarray, k: int)-> Union[int, None]:
    """Quick Select sin recursion de cola

    Args:
        t (np.ndarray): Lista de elementos
        k (int): Indice a recuperar

    Returns:
        Union[int, None]: Elemento en la posicion k o None
    """
    if len(t) == 1 and k == 0:
        return t[0]
    if len(t) == 0:
        return None
    

    m = len(t)
    t_aux = t

    while True:
        t_l, mid, t_r = split(t_aux)
        m = len(t_l)
        
        if k == m:
            return mid
        elif k < m:
            t_aux = t_l
        else:
            t_aux = t_r
            k = k-m-1
        

def split_pivot(t: np.ndarray, mid: int)-> Tuple[np.ndarray, int, np.ndarray]:
    """Divide una lista en menores y mayores del pivot dado

    Args:
        t (np.ndarray): Lista a dividir
        mid (int): pivot

    Returns:
        Tuple[np.ndarray, int, np.ndarray]: Lista de menores, pivot y lista de mayores, en ese orden
    """
    menores = np.array([elemento for elemento in t if elemento < mid])
    mayores = np.array([elemento for elemento in t if elemento > mid])
    return (menores, mid, mayores)

def pivot5(t: np.ndarray)-> int:
    """Calcula el pivote 5, la mediana de medianas de 5 elementos

    Args:
        t (np.ndarray): Lista a recuperar el pivot 5

    Returns:
        int: pivot 5
    """
    t_aux = [t[x:x+5] for x in range(0, len(t), 5)]
    medians = [sorted(x)[len(x)//2] for x in t_aux]
    mid = sorted(medians)[len(medians)//2]
    return mid

def qsel5_nr(t: np.ndarray, k: int)-> Union[int, None]:
    """Quick select utilizando pivot 5 sin recursion de cola

    Args:
        t (np.ndarray): Lista de elementos
        k (int): posicion del elemento a recuperar

    Returns:
        Union[int, None]: Elemento en la posicion k o None 
    """
    mid = pivot5(t)
    t_aux = t
    m = len(t)
    
    while True:
        mid = pivot5(t_aux)
        t_l, mid, t_r = split_pivot(t_aux, mid)
        m = len(t_l)

        if k == m:
            return mid
        elif k < m:
            t_aux = t_l
        else:
            t_aux = t_r
            k = k-m-1

def qsort_5(t: np.ndarray)-> np.ndarray:
    """Ordena la lista dada usando pivot 5 y el metodo quicksort

    Args:
        t (np.ndarray): Lista desordenada

    Returns:
        np.ndarray: Lista ordenada
    """
    if len(t) <= 1:
        return t
    
    pivot = pivot5(t)
    t_l, _, t_r = split_pivot(t, pivot)
    
    sorted_left = qsort_5(t_l)
    sorted_right = qsort_5(t_r)
    
    return np.concatenate([sorted_left, [pivot], sorted_right])

def knapsack_fract_greedy(l_weights: List[int], l_values: List[int], bound: int)-> Dict:
    """Resuleve el problema de la mochila fraccionado

    Args:
        l_weights (List[int]): Pesos de los elementos
        l_values (List[int]): Valor de los elementos
        bound (int): Limite de peso

    Returns:
        Dict: valor del elemento -> cantidad del elemento
    """

    rel_vals = [(i, l_values[i]/l_weights[i]) for i in range(len(l_values))]

    rel_vals = sorted(rel_vals,key=lambda x: x[1], reverse=True)
    elementos_mochila = {elemento: 0 for elemento in l_values}

    ctr = 0
    while bound > 0 and len(l_values) >= ctr:
        idx, _ = rel_vals[ctr]

        if l_weights[idx] <= bound:
            bound -= l_weights[idx]
            elementos_mochila[l_values[idx]] = 1
        else:
            add = bound / l_weights[idx]
            bound -= add * l_weights[idx]
            elementos_mochila[l_values[idx]] = add
        ctr += 1
    
    return elementos_mochila
    

def knapsack_01_pd(l_weights: List[int], l_values: List[int], bound: int)-> int:
    """Resuelve el problema de la mochila 0/1

    Args:
        l_weights (List[int]): Lista de pesos
        l_values (List[int]): Lista de valores asociados a los pesos
        bound (int): Limite de peso

    Returns:
        int: Valor optimo de la mochila
    """
    n = len(l_weights)
    
    dp = [[0] * (bound + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(bound + 1):
            if l_weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], l_values[i - 1] + dp[i - 1][w - l_weights[i - 1]])
            else:
                dp[i][w] = dp[i - 1][w]
    
    return dp[n][bound]

def change_pd(c: int, l_coins: List[int]) -> np.ndarray:
    """ Calcula la matriz generada para hallar el numero de monedas dando cambio de una cantidad c

    Args:
        c (int): Cantidad
        l_coins (List[int]): Lista de monedas

    Returns:
        np.ndarray: Matriz PD
    """
    
    arr = np.full((len(l_coins) + 1, c + 1), np.inf)
    
    
    arr[:, 0] = 0
    
    
    for i in range(1, len(l_coins) + 1):
        for j in range(1, c + 1):
            
            if l_coins[i - 1] <= j:
                
                arr[i][j] = min(arr[i - 1][j], 1 + arr[i][j - l_coins[i - 1]])
            else:
                
                arr[i][j] = arr[i - 1][j]
    
    return arr

def optimal_change_pd(c: int, l_coins: List[int])-> Dict:
    """Calcula el cambio optimo de cambio de monedas dada una cantidad c

    Args:
        c (int): Cantidad
        l_coins (List[int]): Lista de monedas

    Returns:
        Dict: Valor de la moneda -> cantidad de esa moneda
    """

    arr = change_pd(c, l_coins)
    sol = int(arr[-1,-1])
    monedas = {elemento: 0 for elemento in l_coins}

    for _ in range(0, sol):
        for i in range(1, len(arr[:,-1])):
            if arr[-i,-1] < min(arr[:-i,-1]):
                monedas[l_coins[-i]] += 1
                c = c-l_coins[-i]
                arr = change_pd(c, l_coins)
                break
    
    return dict(OrderedDict(sorted(monedas.items())))
