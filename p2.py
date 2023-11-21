import numpy as np
import random
from typing import List
from time import time
from itertools import permutations


def init_cd(n: int):
    return np.full(n, -1, dtype=int)


def union(rep_1: int, rep_2: int, p_cd: np.ndarray):
    if p_cd[rep_1] < p_cd[rep_2]:
        p_cd[rep_2] = rep_1
        return rep_1
    elif p_cd[rep_1] > p_cd[rep_2]:
        p_cd[rep_1] = rep_2
        return rep_2
    else:
        p_cd[rep_2] = rep_1
        p_cd[rep_1] -= 1
        return rep_2


def find(ind: int, p_cd: np.ndarray):
    root = ind

    while p_cd[root] >= 0:
        root = p_cd[root]

    while p_cd[ind] >= 0:
        a = p_cd[ind]
        p_cd[ind] = root
        ind = a
    return root


from queue import PriorityQueue


def create_pq(n: int, l_g: list):
    pq = PriorityQueue()

    for u, v, w in l_g:
        pq.put((w, (u, v)))

    return pq


def kruskal(n: int, l_g: list):
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

    # ctr = 0
    # for i in range(0, len(ds)-1):
    #     if ds[i] < 0:
    #         ctr +=1
    #     if ctr >=2:
    #         return None

    return (n, l_t)


def complete_graph(n_nodes: int, max_weight=50) -> tuple[int, list]:
    n = n_nodes
    l_g = []

    while n_nodes >= 1:
        for i in range(n_nodes - 1):
            l_g.append((i, n_nodes - 1, random.randint(1, max_weight)))

        n_nodes -= 1

    return (n, l_g)


def time_kruskal(n_graphs: int, n_nodes_ini: int, n_nodes_fin: int, step: int) -> List:
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


def kruskal_2(n: int, l_g: list):
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
    """ """
    m = np.random.randint(1, w_max + 1, (n_nodes, n_nodes))
    m = (m + m.T) // 2
    np.fill_diagonal(m, 0)
    return m


def greedy_tsp(dist_m: np.ndarray, node_ini=0) -> List:
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
    dist = 0
    city_before = circuit[0]
    for city in circuit:
        dist += dist_m[city][city_before]
        city_before = city
    return dist


def repeated_greedy_tsp(dist_m: np.ndarray) -> List:
    best_circuit = greedy_tsp(dist_m, 0)

    for city in range(1, len(dist_m[0]) - 1):
        circuit = greedy_tsp(dist_m, city)
        if min(
            len_circuit(circuit, dist_m), len_circuit(best_circuit, dist_m)
        ) == len_circuit(circuit, dist_m):
            best_circuit = circuit
    return best_circuit


def exhaustive_tsp(dist_m: np.ndarray) -> List:
    best_circuit = [item for item in range(0, dist_m.shape[0])]
    for circuit in permutations(range(0, dist_m.shape[0])):
        if min(
            len_circuit(circuit, dist_m), len_circuit(best_circuit, dist_m)
        ) == len_circuit(circuit, dist_m):
            best_circuit = circuit
    return best_circuit
