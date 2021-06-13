from .helpers import dc_eq, normalize

import numpy as np
from typing import Dict, Any
import tensorflow as tf

from dataclasses import dataclass

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

    def get_tf_input_assuming_params0(self) -> Dict[str, np.array]:
        return {
            "wt": self.wt,
            "b": self.b,
            "sig2": self.sig2
            }

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