import pandas as pd
import numpy as np
from typing import List

class ImportHelper:

    @staticmethod
    def import_gillespie_ssa(fnames: str, time: float, species: List[str]) -> np.array:
        # Read first fname
        ff = pd.read_csv(fnames[0], sep=" ")

        # Find row
        times = ff['t'].to_numpy()
        idxs = np.where(times == time)[0]
        if len(idxs) != 1:
            raise ValueError("Could not find time: %f in the data" % time)
        # Add 1 for the header
        idx = idxs[0] + 1
        skiprows = list(np.arange(1,idx))

        # Data to return
        ret = np.zeros(shape=(len(fnames),len(species)))

        # Import
        for i,fname in enumerate(fnames):
            ff = pd.read_csv(fname, skiprows=skiprows, nrows=1, header=0, sep=" ")
            ret[i] = ff[species].to_numpy()[0]
        
        return ret
