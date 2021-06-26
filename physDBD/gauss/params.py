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

class ParamsGaussTE:
    pass

class ParamsGauss:
    pass

@dataclass(eq=False)
class ParamsGaussLF:

    lf: Dict[str,float]
    _nv: int
    _nh: int
    
    def __init__(self, nv: int, nh: int, lf: Dict[str,float]):
        self.lf = lf

        self._nv = nv
        self._nh = nh

    @property
    def nv(self) -> int:
        """No. visible species

        Returns:
            int: No. visible species
        """
        return self._nv

    @property
    def nh(self) -> int:
        """No. hidden species

        Returns:
            int: No. hidden species
        """
        return self._nh

    @property
    def n(self) -> int:
        """No. species
        
        Returns:
            int: No. species
        """
        return self._nv + self._nh
    
    @classmethod
    def fromParamsGauss(cls, params: ParamsGauss):
        lf = {}
        
        for i in range(0,params.n):
            s = "mu_%d" % i
            lf[s] = params.mu[i]

        for i in range(0,params.n):
            for j in range(0,i):
                s = "chol_%d_%d" % (i,j)
                lf[s] = params.chol[i,j]
        
        return cls(
            nv=params.nv, 
            nh=params.nh, 
            lf=lf
            )

@dataclass(eq=False)
class ParamsGauss:

    mu: np.array
    chol: np.array
    
    _nv: int
    _nh: int
    
    def __init__(self, nv: int, nh: int, mu: np.array, chol: np.array):
        self._nv = nv
        self._nh = nh

        self.mu = mu
        self.chol = chol

    @classmethod
    def fromParamsGaussLF(cls, paramsLF: ParamsGaussLF):
        
        mu = np.zeros(paramsLF.n)
        chol = np.zeros((paramsLF.n,paramsLF.n))

        for key,val in paramsLF.lf.items():
            s = key.split('_')
            
            if s[0] == "mu" and s[1].isdigit():
                mu[int(s[1])] = val
            elif s[0] == "chol" and s[1].isdigit() and s[2].isdigit():
                chol[int(s[1]),int(s[2])] = val

        return cls(
            nv=paramsLF.nv,
            nh=paramsLF.nh,
            mu=mu,
            chol=chol
            )

    @property
    def nv(self) -> int:
        """No. visible species

        Returns:
            int: No. visible species
        """
        return self._nv

    @property
    def nh(self) -> int:
        """No. hidden species

        Returns:
            int: No. hidden species
        """
        return self._nh

    @property
    def n(self) -> int:
        """No. species
        
        Returns:
            int: No. species
        """
        return self._nv + self._nh

    def __eq__(self, other):
        return dc_eq(self, other)

    @property
    def prec(self) -> np.array:
        return np.dot(self.chol, np.transpose(self.chol))

    @property
    def cov(self) -> np.array:
        return np.linalg.inv(self.prec)

    def get_tf_input_assuming_params0(self, tpt: int) -> Dict[str, np.array]:
        """Get TF input assuming these are std. params with muh=0, varh = I

        Args:
            tpt (int): Timepoint (not real time)

        Returns:
            Dict[str, np.array]: Keys = "tpt", "mu", "chol"; values are the arrays/floats
        """
        return {
            "tpt": np.array([tpt]).astype(float),
            "mu": np.array([self.mu]),
            "chol": np.array([self.chol])
            }

    @classmethod
    def addParamsGaussLF(cls, params, paramsLF: ParamsGaussLF):

        mu = np.array(params.mu)
        chol = np.array(params.chol)

        for key,val in paramsLF.lf.items():
            s = key.split('_')
            
            if s[0] == "mu" and s[1].isdigit():
                mu[int(s[1])] += val
            elif s[0] == "chol" and s[1].isdigit() and s[2].isdigit():
                chol[int(s[1]),int(s[2])] += val

        return cls(
            nv=paramsLF.nv,
            nh=paramsLF.nh,
            mu=mu,
            chol=chol
            )

    @classmethod
    def addTE(cls, params, paramsTE: ParamsGaussTE):
        """Construct by adding time evolution to existing params.

        Args:
            params (ParamsGauss): ParamsGauss
            paramsTE (ParamsGaussTE): Time evolution
        """
        mu = params.mu + paramsTE.mu_TE
        chol = params.chol + paramsTE.chol_TE
        return cls(
            nv=params.nv,
            nh=params.nh,
            mu=mu,
            chol=chol
            )
    
    @classmethod
    def fromDataStd(cls, data: np.array, nh: int):
        nv = data.shape[1]
        mu_h = np.zeros(nh)
        chol_h = np.eye(nh)
        chol_vh = np.zeros((nh,nv))
        return cls.fromData(
            data=data, 
            mu_h=mu_h,
            chol_h=chol_h,
            chol_vh=chol_vh
            )

    @classmethod
    def fromData(cls, data: np.array, mu_h: np.array, chol_h: np.array, chol_vh: np.array):
        """Construct by applying PCA to data

        Args:
            data (np.array): Data matrix of size (no_seeds, no_species)
            muh (np.array): Latent mean of size nh = no. hidden species
            varh_diag (np.array): Latent varh_daig of size nh = no. hidden species
        """

        assert len(mu_h) == chol_h.shape[0]
        assert chol_h.shape[0] == chol_h.shape[1]
        assert chol_vh.shape[1] == data.shape[1]
        assert chol_vh.shape[0] == len(mu_h)

        nv = chol_vh.shape[1]
        nh = chol_vh.shape[0]

        mu_v = np.mean(data,axis=0)
        cov_v = np.cov(data,rowvar=False)

        # Visible part of chol is same as cholesky decomp. of visible part of cov
        chol_v = np.linalg.cholesky(cov_v)

        # Array flatten
        chol = array_flatten_lower_tri(chol_v, chol_vh, chol_h)

        mu = np.concatenate((mu_v,mu_h))        

        return cls(
            nv=nv,
            nh=nh,
            mu=mu,
            chol=chol
            )

    def _make_a_mat(self, chol_vh, chol_h):
        chol_vh_t = np.transpose(chol_vh)
        chol_h_t = np.transpose(chol_h)

        tmp = np.dot(chol_h, chol_h_t) + np.dot(chol_vh, chol_vh_t)
        tmp = np.linalg.inv(tmp)
        return np.eye(self.nv) - np.dot(chol_vh_t, np.dot(tmp, chol_vh))

    def convert_latent_space(self, mu_h_new: np.array, chol_vh_new: np.array, chol_h_new: np.array):
        
        self.mu[self.nv:] = mu_h_new

        chol_vh = self.chol[self.nv:,:self.nv]
        chol_h = self.chol[self.nv:,self.nv:]

        amat = self._make_a_mat(chol_vh, chol_h)
        amat_new = self._make_a_mat(chol_vh_new, chol_h_new)

        chol_a = np.linalg.cholesky(amat)
        chol_a_new = np.linalg.cholesky(amat_new)

        chol_v = self.chol[:self.nv,:self.nv]
        chol_v_new = np.dot(chol_v, np.dot(chol_a, np.linalg.inv(chol_a_new)))

        self.chol = array_flatten_lower_tri(chol_v_new, chol_vh_new, chol_h_new)