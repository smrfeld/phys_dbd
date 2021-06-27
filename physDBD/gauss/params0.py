from .dparams0 import DParams0Gauss
from ..helpers import dc_eq

import numpy as np
from typing import Dict

from dataclasses import dataclass

class Params0Gauss:
    pass

@dataclass(eq=False)
class Params0Gauss:

    mu_v: np.array
    chol_v: np.array
    
    _nv: int
    
    def __init__(self, nv: int, mu_v: np.array, chol_v: np.array):
        self._nv = nv

        self.mu_v = mu_v
        self.chol_v = chol_v

    def to_lf_dict(self):
        lf = {}
        
        for i in range(0,self.nv):
            s = "mu_v_%d" % i
            lf[s] = self.mu_v[i]

        for i in range(0,self.nv):
            for j in range(0,i+1):
                s = "chol_v_%d_%d" % (i,j)
                lf[s] = self.chol_v[i,j]
        
        return lf

    @classmethod
    def fromLFdict(cls, lf: Dict[str,float], nv: int):
        
        mu_v = np.zeros(nv)
        chol_v = np.zeros((nv,nv))

        for key,val in lf.items():
            s = key.split('_')
            
            if s[0] == "mu" and s[1] == "v" and s[2].isdigit():
                mu_v[int(s[2])] = val
            elif s[0] == "chol" and s[1] == "v" and s[2].isdigit() and s[3].isdigit():
                chol_v[int(s[2]),int(s[3])] = val

        return cls(
            nv=nv,
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
    def addParams0Gauss(cls, params0: Params0Gauss, params0_to_add: Params0Gauss):

        mu_v = params0.mu_v + params0_to_add.mu_v
        chol_v = params0.chol_v + params0_to_add.chol_v

        return cls(
            nv=params0.nv,
            mu_v=mu_v,
            chol_v=chol_v
            )

    @classmethod
    def addLFdict(cls, params0: Params0Gauss, lf: Dict[str, float], nv: int):
        params0_to_add = Params0Gauss.fromLFdict(lf=lf, nv=nv)
        return cls.addParams0Gauss(params0, params0_to_add)

    @classmethod
    def addDeriv(cls, params0: Params0Gauss, dparams0: DParams0Gauss):
        """Construct by adding time evolution to existing params.

        Args:
            params0 (Params0Gauss): ParamsGauss
            dparams0 (DParams0Gauss): Time evolution
        """
        mu_v = params0.mu_v + dparams0.dmu_v
        chol_v = params0.chol_v + dparams0.dchol_v
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