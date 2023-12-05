from typing import Tuple, Union

import numpy as np

def split(t: np.ndarray)-> Tuple[np.ndarray, int, np.ndarray]:
    ...

def qsel(t: np.ndarray, k: int)-> Union[int, None]:
    ...

def qsel_nr(t: np.ndarray, k: int)-> Union[int, None]:
    ...

def split_pivot(t: np.ndarray, mid: int)-> Tuple[np.ndarray, int, np.ndarray]:
    ...

def pivot5(t: np.ndarray)-> int:
    ...

def qsel5_nr(t: np.ndarray, k: int)-> Union[int, None]:
    ...

def qsort_5(t: np.ndarray)-> np.ndarray:
    ...