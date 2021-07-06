from .helpers import dc_eq
from dataclasses import dataclass
import numpy as np

from typing import Dict, List

@dataclass(eq=False)
class ParamsTE:

    wt_TE: np.array
    varh_diag_TE: np.array
    b_TE: np.array
    muh_TE: np.array
    sig2_TE: float

    @property
    def nv(self) -> int:
        """No. visible species

        Returns:
            int: No. visible species
        """
        return len(self.b_TE)

    @property
    def nh(self) -> int:
        """No. hidden species

        Returns:
            int: No. hidden species
        """
        return len(self.muh_TE)

    def get_tf_output_assuming_params0(self, 
        non_zero_outputs : List[str] = []
        ) -> Dict[str, float]:
        """Get TF outputs assuming these are std. params with muh=0,varh_diag=1 and muhTE=0,varh_diagTE=0

        Args:
            non_zero_outputs (List[str], optional): Long form wt00_TE, wt01_TE, etc. 
                If provided, only these are returned, else all are. Defaults to [].

        Returns:
            Dict[str, float]: Keys are long form, i.e.
                wt00_TE, wt01_TE, ..., b0_TE, b1_TE, ..., sig2_TE, muh0_TE, muh1_TE, ..., varh_diag0_TE, varh_diag1_TE, .... 
                Values are floats
        """
        out = {}
        if len(non_zero_outputs) == 0:
            for ih in range(0,self.nh):
                for iv in range(0,self.nv):
                    out["wt%d%d_TE" % (ih,iv)] = self.wt_TE[ih,iv]
            for iv in range(0,self.nv):
                out["b%d_TE" % iv] = self.b_TE[iv]
            out["sig2_TE"] = self.sig2_TE
        
        else:
            for s in non_zero_outputs:
                if s[:2] == "wt":
                    ih = int(s[2])
                    iv = int(s[3])
                    out[s] = self.wt_TE[ih,iv]
                elif s[:1] == "b":
                    iv = int(s[1])
                    out[s] = self.b_TE[iv]
                elif s[:4] == "sig2":
                    out[s] = self.sig2_TE

        return out

    def __eq__(self, other):
        return dc_eq(self, other)

    def to_1d_arr(self) -> np.array:
        """Convert to 1D array

        Returns:
            np.array: 1D array of size nv*nh + nv + 1 + 2*nh
        """
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
        """Construct from 1D array

        Args:
            arr (np.array): 1D array of size nv*nh + nv + 1 + 2*nh
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
            wt_TE=wt,
            b_TE=b,
            sig2_TE=sig2,
            muh_TE=muh,
            varh_diag_TE=varh_diag
            )

    def to_lf_dict(self) -> Dict[str,float]:
        """Convert to long form dictionary

        Returns:
            Dict[str, float]: Keys are long form, i.e.
                wt00_TE, wt01_TE, ..., b0_TE, b1_TE, ..., sig2_TE, muh0_TE, muh1_TE, ..., varh_diag0_TE, varh_diag1_TE, .... 
                Values are floats
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
            lf_dict (Dict[str,float]): Keys are long form, i.e.
                wt00_TE, wt01_TE, ..., b0_TE, b1_TE, ..., sig2_TE, muh0_TE, muh1_TE, ..., varh_diag0_TE, varh_diag1_TE, .... 
                Values are floats
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
            wt_TE=wt,
            b_TE=b,
            sig2_TE=sig2,
            muh_TE=muh,
            varh_diag_TE=varh_diag
            )