import numpy as np
import pandas as pd
from dataclasses import astuple
from typing import List, Tuple

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def construct_mat(n: int, non_zero_idx_pairs: List[Tuple[int,int]], mat_non_zero: np.array):
    mat = np.zeros((n,n))
    for i,pair in enumerate(non_zero_idx_pairs):
        mat[pair[0],pair[1]] = mat_non_zero[i]
        mat[pair[1],pair[0]] = mat_non_zero[i]
    return mat

def construct_mat_non_zero(n: int, non_zero_idx_pairs: List[Tuple[int,int]], mat: np.array):
    assert(check_symmetric(mat))

    mat_non_zero = np.zeros(len(non_zero_idx_pairs))
    for i,pair in enumerate(non_zero_idx_pairs):
        mat_non_zero[i] = mat[pair[0],pair[1]]
    return mat_non_zero

def check_non_zero_idx_pairs(n: int, non_zero_idx_pairs: List[Tuple[int,int]]):
    for i in range(0,n):
        if not (i,i) in non_zero_idx_pairs:
            raise ValueError("All diagonal elements must be specified as non-zero.")

    for pair in non_zero_idx_pairs:
        if pair[0] < pair[1]:
            raise ValueError("Only provide lower triangular indexes.")

def convert_np_to_pd(arr_with_times: np.array, nv: int, nh: int) -> pd.DataFrame:
    """Convert a numpy array of wt, b, sig2, muh, varh_diag to a pandas dataframe with named columns

    Args:
        arr_with_times (np.array): Array of size TxN where T is the number of timepoints and 
            N is the size of (wt,b,sig2,muh,varh_diag) = (nv*nh, nv, 1, nh, nh) = nv*nh + nv + 1 + 2*nh
        nv (int): No. visible species
        nh (int): No. hidden species

    Returns:
        pd.DataFrame: Pandas data frame
    """

    # Convert to pandas
    columns = ["t"]
    for ih in range(0,nh):
        for iv in range(0,nv):
            columns += ["wt%d%d" % (ih,iv)]
    for iv in range(0,nv):
        columns += ["b%d" % iv]
    columns += ["sig2"]
    for ih in range(0,nh):
        columns += ["muh%d" % ih]
    for ih in range(0,nh):
        columns += ["varh_diag%d" % ih]

    df = pd.DataFrame(arr_with_times, columns=columns)
    return df

def normalize(vec: np.array) -> np.array:
    """Normalize 1D np arr

    Args:
        vec (np.array): 1D vec to normalize

    Returns:
        np.array: normalized
    """
    return vec / np.sqrt(np.sum(vec**2))

# Equality of np arrays in data class
# https://stackoverflow.com/a/51743960/1427316
def array_safe_eq(a, b) -> bool:
    """Check if a and b are equal, even if they are numpy arrays. 
        Needed for checking equality with @dataclass(eq=False) decorator"""
    if a is b:
        return True
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a.shape == b.shape and np.max(abs(a - b)) < 1e-8
    if isinstance(a,float) and isinstance(b,float):
        return np.max(abs(a-b)) < 1e-8
    try:
        return a == b
    except TypeError:
        return NotImplemented

def dc_eq(dc1, dc2) -> bool:
   """checks if two dataclasses which hold numpy arrays are equal. 
    Needed for checking equality with @dataclass(eq=False) decorator"""
   if dc1 is dc2:
        return True
   if dc1.__class__ is not dc2.__class__:
       return NotImplemented  # better than False
   t1 = astuple(dc1)
   t2 = astuple(dc2)
   return all(array_safe_eq(a1, a2) for a1, a2 in zip(t1, t2))