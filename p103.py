from typing import Tuple
import numpy as np


def matrix_multiplication(m_1: np.ndarray, m_2: np.ndarray) -> np.ndarray:
    """Opera m_1 X m_2 donde m_1 y m_2 son matrices multiplicables

    Args:
        m_1 (np.ndarray): Matriz A
        m_2 (np.ndarray): Matriz B

    Raises:
        Exception: En caso que las matrices no sean multiplicables

    Returns:
        np.ndarray: A x B
    """
    if m_1.shape[1] != m_2.shape[0]:
        raise Exception("Matrices are not compatible")

    n = m_1.shape[1]  # elements in the column/row
    m_result = np.zeros([m_1.shape[0], m_2.shape[1]])

    for row in range(m_1.shape[0]):
        for column in range(m_2.shape[1]):
            for k in range(n):
                m_result[row][column] += m_1[row][k] * m_2[k][column]

    return m_result


def rec_bb(t: list, f: int, l: int, key: int) -> int:
    """Busqueda binaria en la lista t, implementacion recursiva

    Args:
        t (list): lista ordenada
        f (int): primer indice donde buscar en la lista
        l (int): ultimo indice donde buscar en la lista
        key (int): elemento al que buscar n la lista

    Returns:
        int: Indice donde se encuentra key, o None en caso que no este.
    """
    if l < f:
        return None

    root = (f + l) // 2  # este es sería el root para mejor tiempo
    # root = l
    if key < t[root]:
        return rec_bb(t, f, root - 1, key)
    elif key > t[root]:
        return rec_bb(t, root + 1, l, key)

    return root


def bb(t: list, f: int, l: int, key: int) -> int:
    """Busqueda binaria en la lista t, implementacion iterativa

    Args:
        t (list): lista ordenada
        f (int): primer indice donde buscar en la lista
        l (int): ultimo indice donde buscar en la lista
        key (int): elemento al que buscar n la lista

    Returns:
        int: Indice donde se encuentra key, o None en caso que no este.
    """

    while f <= l:
        root = (f + l) // 2  # este es sería el root para mejor tiempo
        # root = l
        if t[root] > key:  #
            l = root - 1
        elif t[root] < key:
            f = root + 1
        elif t[root] == key:
            return root

    return None


def min_heapify(h: np.ndarray, i: int):
    """Heapifica el ndarray h desde el subarbol que empieza en i.

    Args:
        h (np.ndarray): Min heap
        i (int): Nodo desde donde aplicamos Heapify.
    """

    heapifying = True

    while heapifying:
        smallest = i
        left = 2 * i + 1
        right = 2 * i + 2

        if left < len(h) and h[i] > h[left]:
            smallest = left
        if right < len(h) and h[smallest] > h[right]:
            smallest = right

        if smallest != i:
            h[i], h[smallest] = h[smallest], h[i]
            i = smallest
        else:
            heapifying = False


def insert_min_heap(h: np.ndarray, k: int) -> np.ndarray:
    """Inserta el valor k en el array h, conservando el min_heap

    Args:
        h (np.ndarray): min heap
        k (int): valor a insertar

    Returns:
        np.ndarray: min heap con el valor k añadido
    """
    h = np.array(list(h) + [k])
    last = len(h) - 1

    while last > 0 and h[(last - 1) // 2] > h[last]:
        h[(last - 1) // 2], h[last] = h[last], h[(last - 1) // 2]
        last = (last - 1) // 2

    return h


def create_min_heap(h: np.ndarray):
    """Convierte el ndarray h en un min_heap

    Args:
        h (np.ndarray): Lista a convertir a ndarray
    """
    i = ((len(h) - 1) - 1) // 2
    while i > -1:
        min_heapify(h, i)
        i -= 1


def pq_ini() -> np.ndarray:
    """Inicializa una Priority Queue (min heap) vacia

    Returns:
        np.ndarray: Priority queue
    """
    return np.empty(0)


def pq_insert(h: np.ndarray, k: int) -> np.ndarray:
    """Añade el elemento k a la Priority Queue h

    Args:
        h (np.ndarray): Priority Queue
        k (int): elemento a añadir

    Returns:
        np.ndarray: PQ h con el elemento k añadido
    """
    return insert_min_heap(h, k)


def pq_remove(h: np.ndarray) -> Tuple[int, np.ndarray]:
    """Saca de la PQ el elemento con mas prioridad. Como es un min heap, el de menor valor

    Args:
        h (np.ndarray): Priority Queue

    Returns:
        Tuple[int, np.ndarray]: (valor, PQ sin el valor sacado)
    """
    value, h = h[0], h[1:]
    create_min_heap(h)
    return value, h


def min_heap_sort(h: np.ndarray) -> np.ndarray:
    """Utiliza el min heap para ordenar el array h

    Args:
        h (np.ndarray): Array a ordenar

    Returns:
        np.ndarray: Array ordenado.
    """
    create_min_heap(h)
    srtd_list = []

    while len(h) > 0:
        element, h = pq_remove(h)
        srtd_list.append(element)

    return np.array(srtd_list)
