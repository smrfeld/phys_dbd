from .helpers import dc_eq, normalize
from .paramsTE import ParamsTE

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

    def get_tf_input_assuming_params0(self, tpt: int) -> Dict[str, np.array]:
        return {
            "tpt": np.array([tpt]).astype(float),
            "wt": np.array([self.wt]),
            "b": np.array([self.b]),
            "sig2": np.array([self.sig2])
            }

    @classmethod
    def addTE(cls, params, paramsTE: ParamsTE):
        wt = params.wt + paramsTE.wt_TE
        b = params.b + paramsTE.b_TE
        varh_diag = params.varh_diag + paramsTE.varh_diag_TE
        muh = params.muh + paramsTE.muh_TE
        sig2 = params.sig2 + paramsTE.sig2_TE
        return cls(
            wt=wt,
            b=b,
            varh_diag=varh_diag,
            muh=muh,
            sig2=sig2
            )
    
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

    def to_lf_dict(self) -> Dict[str,float]:
        lf_dict = {}

        for ih in range(0,self.nh):
            for iv in range(0,self.nv):
                s = "wt%d%d" % (ih, iv)
                lf_dict[s] = self.wt[ih,iv]
        
        for iv in range(0,self.nv):
            s = "b%d" % iv
            lf_dict[s] = self.b[iv]
        
        s = "sig2"
        lf_dict[s] = self.sig2

        for ih in range(0,self.nh):
            s = "muh%d" % ih
            lf_dict[s] = self.muh[ih]

        for ih in range(0,self.nh):
            s = "varh_diag%d" % ih
            lf_dict[s] = self.varh_diag[ih]

        return lf_dict

    @classmethod
    def fromLFdict(cls, lf_dict: Dict[str,float], nv: int, nh: int):
        wt = np.zeros((nh,nv))
        for ih in range(0,nh):
            for iv in range(0,nv):
                s = "wt%d%d" % (ih,iv)
                wt[ih,iv] = lf_dict[s]

        b = np.zeros(nv)
        for iv in range(0,nv):
            s = "b%d" % iv
            b[iv] = lf_dict[s]

        sig2 = lf_dict["sig2"]

        muh = np.zeros(nh)
        for ih in range(0,nh):
            s = "muh%d" % ih
            muh[ih] = lf_dict[s]

        varh_diag = np.zeros(nh)
        for ih in range(0,nh):
            s = "varh_diag%d" % ih
            varh_diag[ih] = lf_dict[s]
        
        return cls(
            wt=wt,
            b=b,
            sig2=sig2,
            muh=muh,
            varh_diag=varh_diag
            )

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