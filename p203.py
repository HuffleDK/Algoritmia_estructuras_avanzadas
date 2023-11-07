import numpy as np

def init_cd(n: int):
    return np.full(n, -1, dtype=int)

print(init_cd(5))


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

test = np.array = [-4, 0, 5, 1, 0, 4, -3, 6, 7]
#print(union(0, 6, test))

def find(ind: int, p_cd: np.ndarray):
    root = ind

    print(p_cd)
    while p_cd[root] >= 0:
        root  = p_cd[root]
    
    print(f"la raíz de {ind}: {root}")

    while p_cd[ind] >= 0:
        a = p_cd[ind]
        p_cd[ind] = root
        ind = a
    return root
    

	
print(test)
print(find(2, test))

from queue import PriorityQueue

def create_pq(n: int, l_g: list):
    pq = PriorityQueue()

    for u, v, w in l_g:
        pq.put((w, (u,v)))
    
    return pq
    

def kruskal(n: int, l_g: list):
    pq = create_pq(n, l_g)
    ds = init_cd(n)
    l_t = []

    while not pq.empty():
        _, (u, v) = pq.get()

        print(f"Examining edge {u}, {v}")

        x = find(u, ds)
        y = find(v, ds)

        print(f"Got roots {x}, {y}")  
        
        if x != y:
            print("Doing the union")
            l_t.append((u,v))
            union(x, y, ds)
    
    # ctr = 0
    # for i in range(0, len(ds)-1):
    #     if ds[i] < 0:
    #         ctr +=1
    #     if ctr >=2:
    #         return None
        

    return (n, l_t)
import random
from typing import List
def complete_graph(n_nodes: int, max_weight=50)-> tuple[int, list]:
    n = n_nodes
    l_g = []

    while n_nodes >= 1:
        for i in range(n_nodes-1):
            l_g.append((i, n_nodes -1, random.randint(1, max_weight)))

        n_nodes -= 1

    
    return (n, l_g)


n, a = complete_graph(15, 55)

print(a)

kruskal(n, a)


def time_kruskal(n_graphs: int, n_nodes_ini: int, n_nodes_fin: int, step: int)-> List:

    pass