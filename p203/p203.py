import random
from typing import List, Tuple
from time import time
from itertools import permutations
from queue import PriorityQueue

import numpy as np


def init_cd(n: int) -> np.ndarray:
    """Inicializa un conjunto disjunto en forma de array de numpy inicializado a -1

    Args:
        n (int): tamaño del conjunto disjunto

    Returns:
        np.ndarray: Conjunto Disjunto
    """
    return np.full(n, -1, dtype=int)


def union(rep_1: int, rep_2: int, p_cd: np.ndarray) -> int:
    """En el conjunto disjunto p_cd, une los nodos rep_1 y rep_2, 
    creando el arbol con menor altura posible

    Args:
        rep_1 (int): raiz del primer arbol
        rep_2 (int): raiz del segundo arbol
        p_cd (np.ndarray): Conjunto disjunto
    
    Returns:
        int: raiz del arbol union resultado
    """
    if p_cd[rep_1] < p_cd[rep_2]:
        p_cd[rep_2] = rep_1
        return rep_1
    if p_cd[rep_1] > p_cd[rep_2]:
        p_cd[rep_1] = rep_2
        return rep_2
    p_cd[rep_2] = rep_1
    p_cd[rep_1] -= 1
    return rep_2


def find(ind: int, p_cd: np.ndarray) -> int:
    """Encuentra la raiz del nodo ind en p_cd y aplica compresión de caminos

    Args:
        ind (int): indice que encontrar
        p_cd (np.ndarray): Conjunto disjunto

    Returns:
        int: raiz del nodo ind
    """
    root = ind

    while p_cd[root] >= 0:
        root = p_cd[root]

    while p_cd[ind] >= 0:
        a = p_cd[ind]
        p_cd[ind] = root
        ind = a
    return root


def create_pq(n: int, l_g: List) -> PriorityQueue:
    """Crea una cola de prioridad dado una lista de aristas. 
    Las aristas deben estar formateadas de forma (nodo1, nodo2, distancia entre ambos)

    Args:
        n (int): Numero de nodos en el grafo
        l_g (List): Lista con aristas.
    Returns:
        PriorityQueue: Cola de prioridad
    """
    pq = PriorityQueue()

    for u, v, w in l_g:
        pq.put((w, (u, v)))

    return pq


def kruskal(n: int, l_g: List) -> Tuple[int, List]:
    """Dado un grafo en forma de lista de aristas, encuentra el arbol recubridor minimo

    Args:
        n (int): Numero de vertices en el grafo 
        l_g (int): Lista de aristas que compone el grafo
    Returns:
        Tuple[int, List]: Numero de nodos del grafo, lista de aristas final.
    """
    pq = create_pq(n, l_g)
    ds = init_cd(n)
    l_t = []

    while not pq.empty():
        _, (u, v) = pq.get()

        x = find(u, ds)
        y = find(v, ds)

        if x != y:
            l_t.append((u, v))
            union(x, y, ds)

    return (n, l_t)


def complete_graph(n_nodes: int, max_weight: int=50) -> Tuple[int, List]:
    """Genera un grafo completo con n_nodes nodos.

    Args:
        n_nodes (int): Numero de nodos del grafo a generar.
        max_weight (optional, int): Máximo peso de las aristas. Por defecto 50.
    Returns:
        (int, List): Numero de nodos del grafo, lista con las aristas. 
    """
    n = n_nodes
    l_g = []

    while n_nodes >= 1:
        for i in range(n_nodes - 1):
            l_g.append((i, n_nodes - 1, random.randint(1, max_weight)))

        n_nodes -= 1

    return (n, l_g)


def time_kruskal(n_graphs: int, n_nodes_ini: int, n_nodes_fin: int, step: int) -> List:
    """Calcula el tiempo de ejecucción de la función kruskal. 
    Genera varios grafos, aumentado el tamaño.

    Args:
        n_graphs (int): Numero de grafos generados en cada timing.
        n_nodes_ini (int): Tamaño del grafo inicial.
        n_nodes_fin (int): Máximo tamaño del grafo.
        step (int): Cantidad de nodos a añadir con cada timing.
    Returns:
        List: Lista de tiempos de ejecución con el tamaño de cada grafo
    """
    times = []

    for n in range(n_nodes_ini, n_nodes_fin + 1, step):
        l_graphs = []
        l_graphs = [complete_graph(n, 50) for i in range(0, n_graphs)]
        media = 0
        for graph in l_graphs:
            time1 = time()
            kruskal(graph[0], graph[1])
            time2 = time()
            media = media + time2 - time1
        media = media / n_graphs
        times.append((n, media))

    return times


def kruskal_2(n: int, l_g: list) -> Tuple[int, List, float]:
    """Funcion auxiliar. Ademas de ejecutar kruskal, 
    devuelve el tiempo de gestion del conjunto disjunto 

    Args:
        n (int): Numero de vertices en el grafo 
        l_g (int): Lista de aristas que compone el grafo
    Returns:
        Tuple[int, List, float]: Numero de nodos del grafo, lista de aristas final.
    """

    pq = create_pq(n, l_g)
    time1 = time()
    ds = init_cd(n)
    time1_fin = time()
    l_t = []

    time_k = time1_fin - time1
    while not pq.empty():
        _, (u, v) = pq.get()

        time2 = time()
        x = find(u, ds)
        y = find(v, ds)
        time2_fin = time()

        time3, time3_fin = 0, 0
        if x != y:
            l_t.append((u, v))
            time3 = time()
            union(x, y, ds)
            time3_fin = time()

        time_k += time2_fin - time2 + time3_fin - time3

    return (n, l_t, time_k)


def time_kruskal_2(
    n_graphs: int, n_nodes_ini: int, n_nodes_fin: int, step: int
) -> List:
    """Calcula el tiempo de ejecucción de la función kruskal 
    pero solo los tiempos de gestion de conjunto disjunto. 
    Genera varios grafos, aumentado el tamaño.

    Args:
        n_graphs (int): Numero de grafos generados en cada timing.
        n_nodes_ini (int): Tamaño del grafo inicial.
        n_nodes_fin (int): Máximo tamaño del grafo.
        step (int): Cantidad de nodos a añadir con cada timing.
    Returns:
        List: Lista de tiempos de ejecución con el tamaño de cada grafo
    """
    times = []

    for n in range(n_nodes_ini, n_nodes_fin + 1, step):
        l_graphs = []
        l_graphs = [complete_graph(n, 50) for i in range(0, n_graphs)]
        media = 0
        for graph in l_graphs:
            _, _, k_time = kruskal_2(graph[0], graph[1])
            media = media + k_time
        media = media / n_graphs
        times.append((n, media))

    return times


def dist_matrix(n_nodes: int, w_max=10) -> np.ndarray:
    """Genera una matriz de distancias aleatoria.
    
    Args:
        n_nodes (int): Numero de nodos en el grafo.
        w_max (int, opcional): Máxima distancia entre nodos, por defecto 10.
    Returns:
        np.ndarray: Matriz de tamaño n_nodes x n_nodes
    """
    m = np.random.randint(1, w_max + 1, (n_nodes, n_nodes))
    m = (m + m.T) // 2
    np.fill_diagonal(m, 0)
    return m


def greedy_tsp(dist_m: np.ndarray, node_ini=0) -> List:
    """Placeholder for gtsp"""
    num_cities = dist_m.shape[0]
    circuit = [node_ini]
    while len(circuit) < num_cities:
        current_city = circuit[-1]
        # sort cities in ascending distance from current
        options = np.argsort(dist_m[current_city])
        # add first city in sorted list not visited yet
        for city in options:
            if city not in circuit:
                circuit.append(city)
                break
    return np.array(circuit)


def len_circuit(circuit: List, dist_m: np.ndarray) -> int:
    """Dado un circuito devuelve su distancia
    """
    dist = 0
    city_before = circuit[0]
    for city in circuit:
        dist += dist_m[city][city_before]
        city_before = city
    return dist


def repeated_greedy_tsp(dist_m: np.ndarray) -> List:
    """Placeholder"""
    best_circuit = greedy_tsp(dist_m, 0)

    for city in range(1, len(dist_m[0]) - 1):
        circuit = greedy_tsp(dist_m, city)
        if min(
            len_circuit(circuit, dist_m), len_circuit(best_circuit, dist_m)
        ) == len_circuit(circuit, dist_m):
            best_circuit = circuit
    return best_circuit


def exhaustive_tsp(dist_m: np.ndarray) -> List:
    """Placeholder"""
    best_circuit = [item for item in range(0, dist_m.shape[0])]
    for circuit in permutations(range(0, dist_m.shape[0])):
        if min(
            len_circuit(circuit, dist_m), len_circuit(best_circuit, dist_m)
        ) == len_circuit(circuit, dist_m):
            best_circuit = circuit
    return best_circuit
