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
    def nv(self) -> int:
        """No. visible species

        Returns:
            int: No. visible species
        """
        return len(self.b)

    @property
    def nh(self) -> int:
        """No. hidden species

        Returns:
            int: No. hidden species
        """
        return len(self.muh)

    def __eq__(self, other):
        return dc_eq(self, other)

    def get_tf_input_assuming_params0(self, tpt: int) -> Dict[str, np.array]:
        """Get TF input assuming these are std. params with muh=0, varh_diag=1

        Args:
            tpt (int): Timepoint (not real time)

        Returns:
            Dict[str, np.array]: Keys = "tpt", "wt", "b", "sig2"; values are the arrays/floats
        """
        return {
            "tpt": np.array([tpt]).astype(float),
            "wt": np.array([self.wt]),
            "b": np.array([self.b]),
            "sig2": np.array([self.sig2])
            }

    @classmethod
    def addLFdict(cls, params, lf_dict: Dict[str,float]):
        """Construct new parameters by adding a long form (lf) dictionary of time evolutions

        Args:
            params (Params): Params
            lf_dict (Dict[str,float]): Long form dictionary to add. Keys should be:
                wt00_TE, wt01_TE, ..., b0_TE, b1_TE, ..., sig2_TE, muh0_TE, muh1_TE, ..., varh_diag0_TE, varh_diag1_TE, ...
        """
        wt = np.array(params.wt)
        sig2 = params.sig2
        b = np.array(params.b)
        muh = np.array(params.muh)
        varh_diag = np.array(params.varh_diag)
        for lf, val in lf_dict.items():
            if lf[:2] == "wt":
                ih = int(lf[2])
                iv = int(lf[3])
                wt[ih,iv] += val
            elif lf[:1] == "b":
                iv = int(lf[1])
                b[iv] += val
            elif lf[:4] == "sig2":
                sig2 += val
            elif lf[:3] == "muh":
                ih = int(lf[3])
                muh[ih] += val
            elif lf[:4] == "varh":
                ih = int(lf[-1])
                varh_diag[ih] += val
        
        return cls(
            wt=wt,
            b=b,
            varh_diag=varh_diag,
            muh=muh,
            sig2=sig2
            )

    @classmethod
    def addTE(cls, params, paramsTE: ParamsTE):
        """Construct by adding time evolution to existing params.

        Args:
            params (Parmas): Params
            paramsTE (ParamsTE): Time evolution
        """
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
        """Construct by applying PCA to data

        Args:
            data (np.array): Data matrix of size (no_seeds, no_species)
            muh (np.array): Latent mean of size nh = no. hidden species
            varh_diag (np.array): Latent varh_daig of size nh = no. hidden species
        """

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
        """Convert params in the latent space to a new muh,varh_diag

        Args:
            muh_new (np.array): New muh of length nh = no. hidden species
            varh_diag_new (np.array): New varh_diag of length nh = no. hidden species
        """

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
        """Convert to long form dictionary

        Returns:
            Dict[str,float]: Keys are wt00, wt01, ..., b0, b1, ..., sig2, muh0, muh1, ..., varh_diag0, varh_diag1, .... Values are floats.
        """
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
        """Construct from long form dictionary

        Args:
            lf_dict (Dict[str,float]): Keys are wt00, wt01, ..., b0, b1, ..., sig2, muh0, muh1, ..., varh_diag0, varh_diag1, .... 
                Values are floats.
            nv (int): No. visible species
            nh (int): No. hidden species
        """
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
        """Convert to 1D numpy array

        Returns:
            np.array: 1D numpy array of (wt,b,sig2,muh,varh_diag) flattened
        """
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
        """Construct from 1D numpy array

        Args:
            arr (np.array): (wt,b,sig2,muh,varh_diag) flattened
            nv (int): No. visible species
            nh (int): No. hidden species
        """
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