from .data_desc import DataDesc

import numpy as np
import pandas as pd

from dataclasses import dataclass, astuple

from typing import List,Tuple

def normalize(vec: np.array) -> np.array:
    return vec / np.sqrt(np.sum(vec**2))

# Equality of np arrays in data class
# https://stackoverflow.com/a/51743960/1427316
def array_safe_eq(a, b) -> bool:
    """Check if a and b are equal, even if they are numpy arrays"""
    if a is b:
        return True
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a.shape == b.shape and np.max(abs(a - b)) < 1e-8
    try:
        return a == b
    except TypeError:
        return NotImplemented

def dc_eq(dc1, dc2) -> bool:
   """checks if two dataclasses which hold numpy arrays are equal"""
   if dc1 is dc2:
        return True
   if dc1.__class__ is not dc2.__class__:
       return NotImplemented  # better than False
   t1 = astuple(dc1)
   t2 = astuple(dc2)
   return all(array_safe_eq(a1, a2) for a1, a2 in zip(t1, t2))

@dataclass(eq=False)
class Params:

    wt: np.array
    varh_diag: np.array
    b: np.array
    muh: np.array
    sig2: float

    @property
    def nv(self):
        return len(self.b)

    @property
    def nh(self):
        return len(self.muh)

    def __eq__(self, other):
        return dc_eq(self, other)

    @classmethod
    def fromPCA(cls, data: np.array, muh: np.array, varh_diag: np.array):

        if len(muh) != len(varh_diag):
            raise ValueError("muh and varh_diag must be of the same length")

        # Dimensionality of data
        d = data.shape[1]

        # Number of latent parameters q < d
        q = len(muh)
        if q >= d:
            raise ValueError("Number of latent parameters: %d muss be less than dimensionality of the data: %d" % (q,d))

        # Mean
        b_ml = np.mean(data,axis=0)

        # Cov
        data_cov = np.cov(np.transpose(data))

        # Eigenvals/vecs, normalized by np
        eigenvals, eigenvecs = np.linalg.eig(data_cov)

        # Adjust sign
        direction_goal = normalize(np.ones(d))
        for i in np.arange(0,len(eigenvecs)):
            angle = np.arccos(np.dot(direction_goal,eigenvecs[:,i]))
            if abs(angle) < 0.5 * np.pi:
                eigenvecs[:,i] *= -1
        
        var_ml = (1.0 / (d-q)) * np.sum(eigenvals[q:d])
        uq = eigenvecs[:,:q]
        eigenvalsq = np.diag(eigenvals[:q])
        weight_ml = np.linalg.multi_dot([uq, np.sqrt(eigenvalsq - var_ml * np.eye(q))])

        # Make params in standardized space
        muh_0 = np.full(q, 0.0)
        varh_diag_0 = np.ones(q)
        params = Params(
            wt=np.transpose(weight_ml),
            b=b_ml,
            varh_diag=varh_diag_0,
            muh=muh_0,
            sig2=var_ml
            )

        # Convert params in standardized space
        params.convert_latent_space(muh, varh_diag)

        return params

    def convert_latent_space(self, muh_new: np.array, varh_diag_new: np.array):

        b1 = self.b
        wt1 = self.wt
        muh1 = self.muh
        varh1 = np.diag(self.varh_diag)
        varh1_sqrt = np.sqrt(varh1)

        muh2 = muh_new
        varh2 = np.diag(varh_diag_new)

        w1 = np.transpose(wt1)
        varh2_inv_sqrt = np.linalg.inv(np.sqrt(varh2))

        b2 = b1 + np.dot(w1,muh1) - np.linalg.multi_dot([w1,varh1_sqrt,varh2_inv_sqrt,muh2])
        wt2 = np.linalg.multi_dot([varh2_inv_sqrt,varh1_sqrt,wt1])

        self.b = b2
        self.wt = wt2

    def to_1d_arr(self) -> np.array:
        x = np.concatenate([
            self.wt.flatten(),
            self.b,
            np.array([self.sig2]),
            self.muh,
            self.varh_diag
            ])
        return x.flatten()

    @classmethod
    def from1dArr(cls, arr: np.array, nv: int, nh: int):
        s = 0
        e = s + nv*nh
        wt_flat = arr[s:e]
        wt = np.reshape(wt_flat,newshape=(nh,nv))

        s = e
        e = s + nv
        b = arr[s:e]

        s = e
        e = s + 1
        sig2 = arr[s:e][0]

        s = e
        e = s + nh
        muh = arr[s:e]

        s = e
        e = s + nh
        varh_diag = arr[s:e]

        return cls(
            wt=wt,
            b=b,
            sig2=sig2,
            muh=muh,
            varh_diag=varh_diag
            )

def convert_params_traj_to_pd(times: np.array, params_traj: List[Params]) -> pd.DataFrame:

    # Get length of 1D representation
    l = len(params_traj[0].to_1d_arr())

    # Convert to np array
    arr = np.zeros(shape=(len(params_traj),l))
    for i,params in enumerate(params_traj):
        arr[i] = params.to_1d_arr()

    # Add times
    arr = np.transpose(np.concatenate((np.array([times]),np.transpose(arr))))

    # Convert to pandas
    nv = params_traj[0].nv
    nh = params_traj[0].nh
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

    df = pd.DataFrame(arr, columns=columns)
    return df

def export_params_traj(fname: str, times: np.array, params_traj: List[Params]):
    
    # Convert to pandas
    df = convert_params_traj_to_pd(times, params_traj)

    # Export pandas
    df.to_csv(fname, sep=" ")

def import_params_traj(fname: str, nv: int, nh: int) -> Tuple[np.array,List[Params]]:

    # Import
    df = pd.read_csv(fname, sep=" ")

    # To numpy
    arr = df.to_numpy()

    params_traj = []
    times = []
    for arr1d in arr:
        t = arr1d[1]
        times.append(t)

        arr1d0 = arr1d[2:]
        params = Params.from1dArr(arr1d0, nv, nh)
        params_traj.append(params)

    return (np.array(times),params_traj)