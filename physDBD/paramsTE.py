from .helpers import dc_eq
from dataclasses import dataclass
import numpy as np

# import tensorflow as tf

from typing import Dict

@dataclass(eq=False)
class ParamsTE:

    wt_TE: np.array
    varh_diag_TE: np.array
    b_TE: np.array
    muh_TE: np.array
    sig2_TE: float

    @property
    def nv(self):
        return len(self.b_TE)

    @property
    def nh(self):
        return len(self.muh_TE)

    def get_tf_output_assuming_params0(self) -> Dict[str, np.array]:
        return {
            "wt_TE": self.wt_TE,
            "b_TE": self.b_TE,
            "sig2_TE": self.sig2_TE
            }

    def __eq__(self, other):
        return dc_eq(self, other)

    def to_1d_arr(self) -> np.array:
        x = np.concatenate([
            self.wt_TE.flatten(),
            self.b_TE,
            np.array([self.sig2_TE]),
            self.muh_TE,
            self.varh_diag_TE
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
            wt_TE=wt,
            b_TE=b,
            sig2_TE=sig2,
            muh_TE=muh,
            varh_diag_TE=varh_diag
            )

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
            wt_TE=wt,
            b_TE=b,
            sig2_TE=sig2,
            muh_TE=muh,
            varh_diag_TE=varh_diag
            )