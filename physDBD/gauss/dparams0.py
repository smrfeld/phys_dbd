from ..helpers import dc_eq

import numpy as np
from typing import Dict, List

from dataclasses import dataclass

@dataclass(eq=False)
class DParams0Gauss:

    dmu_v: np.array
    dchol_v: np.array
    
    _nv: int
    
    def __init__(self, nv: int, dmu_v: np.array, dchol_v: np.array):
        self._nv = nv

        self.dmu_v = dmu_v
        self.dchol_v = dchol_v

    def get_tf_output(self, 
        non_zero_outputs : List[str] = []
        ) -> Dict[str, np.array]:
        out = {}
        if len(non_zero_outputs) == 0:
            for i in range(0,self.nv):
                out["dmu_v_%d" % i] = self.dmu_v[i]
            for i in range(0,self.nv):
                for j in range(0,i+1):
                    out["dchol_v_%d_%d" % (i,j)] = self.dchol_v[i,j]
        
        else:
            for s in non_zero_outputs:
                if s[:6] == "chol_v":
                    i = int(s[7])
                    j = int(s[9])
                    out[s] = self.dchol_v[i,j]
                elif s[:4] == "mu_v":
                    i = int(s[5])
                    out[s] = self.dmu_v[i]

        return out
    
    def to_lf_dict(self):
        lf = {}
        
        for i in range(0,self.nv):
            s = "dmu_v_%d" % i
            lf[s] = self.dmu_v[i]

        for i in range(0,self.nv):
            for j in range(0,i+1):
                s = "dchol_v_%d_%d" % (i,j)
                lf[s] = self.dchol_v[i,j]
        
        return lf

    @classmethod
    def fromLFdict(cls, lf: Dict[str,float], nv: int):
        
        dmu_v = np.zeros(nv)
        dchol_v = np.zeros((nv,nv))

        for key,val in lf.items():
            s = key.split('_')
            
            if s[0] == "dmu" and s[1] == "v" and s[2].isdigit():
                dmu_v[int(s[2])] = val
            elif s[0] == "dchol" and s[1] == "v" and s[2].isdigit() and s[3].isdigit():
                dchol_v[int(s[2]),int(s[3])] = val

        return cls(
            nv=nv,
            dmu_v=dmu_v,
            dchol_v=dchol_v
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