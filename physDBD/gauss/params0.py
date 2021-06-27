from ..helpers import dc_eq

import numpy as np
from typing import Dict

from dataclasses import dataclass

def array_flatten_lower_tri(a, lt, d):
    zeros = np.zeros(np.transpose(lt).shape)
    return array_flatten_sym(a, zeros, lt, d)

def array_flatten_sym(a, b, c, d):
    top = np.concatenate((a,b),axis=1)
    bottom = np.concatenate((c,d),axis=1)
    return np.concatenate((top,bottom),axis=0)

class Params0GaussTE:
    pass

class Params0Gauss:
    pass

@dataclass(eq=False)
class Params0GaussLF:

    lf: Dict[str,float]
    _nv: int
    
    def __init__(self, nv: int, lf: Dict[str,float]):
        self.lf = lf
        self._nv = nv

    @property
    def nv(self) -> int:
        """No. visible species

        Returns:
            int: No. visible species
        """
        return self._nv

    @classmethod
    def fromParamsGauss(cls, params0: Params0Gauss):
        lf = {}
        
        for i in range(0,params0.nv):
            s = "mu_v_%d" % i
            lf[s] = params0.mu_v[i]

        for i in range(0,params0.nv):
            for j in range(0,i+1):
                s = "chol_v_%d_%d" % (i,j)
                lf[s] = params0.chol_v[i,j]
        
        return cls(
            nv=params0.nv, 
            lf=lf
            )

@dataclass(eq=False)
class Params0Gauss:

    mu_v: np.array
    chol_v: np.array
    
    _nv: int
    
    def __init__(self, nv: int, mu_v: np.array, chol_v: np.array):
        self._nv = nv

        self.mu_v = mu_v
        self.chol_v = chol_v

    @classmethod
    def fromParams0GaussLF(cls, params0LF: Params0GaussLF):
        
        mu_v = np.zeros(params0LF.nv)
        chol_v = np.zeros((params0LF.nv,params0LF.nv))

        for key,val in params0LF.lf.items():
            s = key.split('_')
            
            if s[0] == "mu" and s[1] == "v" and s[2].isdigit():
                mu_v[int(s[2])] = val
            elif s[0] == "chol" and s[1] == "v" and s[2].isdigit() and s[3].isdigit():
                chol_v[int(s[2]),int(s[3])] = val

        return cls(
            nv=params0LF.nv,
            mu_v=mu_v,
            chol_v=chol_v
            )

    @property
    def nv(self) -> int:
        """No. visible species

        Returns:
            int: No. visible species
        """
        return self._nv

    def __eq__(self, other):
        return dc_eq(self, other)

    @property
    def prec_v(self) -> np.array:
        return np.dot(self.chol_v, np.transpose(self.chol_v))

    @property
    def cov_v(self) -> np.array:
        return np.linalg.inv(self.prec_v)

    def get_tf_input(self, tpt: int) -> Dict[str, np.array]:
        """Get TF input assuming these are std. params with muh=0, varh = I

        Args:
            tpt (int): Timepoint (not real time)

        Returns:
            Dict[str, np.array]: Keys = "tpt", "mu_v", "chol_v"; values are the arrays/floats
        """
        return {
            "tpt": np.array([tpt]).astype(float),
            "mu_v": np.array([self.mu_v]),
            "chol_v": np.array([self.chol_v])
            }

    @classmethod
    def addParams0Gauss(cls, params0: Params0Gauss, params0_to_add: Params0GaussLF):

        mu_v = params0.mu_v + params0_to_add.mu_v
        chol_v = params0.chol_v + params0_to_add.chol_v

        return cls(
            nv=params0.nv,
            mu_v=mu_v,
            chol_v=chol_v
            )

    @classmethod
    def addParamsGaussLF(cls, params0: Params0Gauss, params0LF: Params0GaussLF):
        params0_to_add = Params0Gauss.fromParams0GaussLF(params0LF)
        return cls.addParams0Gauss(params0, params0_to_add)

    @classmethod
    def addTE(cls, params0: Params0Gauss, params0TE: Params0GaussTE):
        """Construct by adding time evolution to existing params.

        Args:
            params0 (Params0Gauss): ParamsGauss
            params0TE (Params0GaussTE): Time evolution
        """
        mu_v = params0.mu_v + params0TE.mu_v_TE
        chol_v = params0.chol_v + params0TE.chol_v_TE
        return cls(
            nv=params0.nv,
            mu_v=mu_v,
            chol_v=chol_v
            )

    @classmethod
    def fromData(cls, data: np.array):
        """Construct by applying PCA to data

        Args:
            data (np.array): Data matrix of size (no_seeds, no_species)
        """

        nv = data.shape[1]

        mu_v = np.mean(data,axis=0)
        cov_v = np.cov(data,rowvar=False)

        # Visible part of chol is same as cholesky decomp. of visible part of cov
        chol_v = np.linalg.cholesky(cov_v)

        return cls(
            nv=nv,
            mu_v=mu_v,
            chol_v=chol_v
            )