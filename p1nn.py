import numpy as np


def matrix_multiplication(m_1: np.ndarray, m_2: np.ndarray) -> np.ndarray:
    if m_1.shape[1] != m_2.shape[0]:
        raise Exception("Matrices are not compatible")
    
    