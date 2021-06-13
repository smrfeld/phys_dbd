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