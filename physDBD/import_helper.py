from .data_desc import DataDesc

import pandas as pd
import numpy as np
from typing import List

import os

class ImportHelper:

    @staticmethod
    def create_fnames(
        data_dir: str, 
        vol_exp: int, 
        no_ip3r: int, 
        ip3_dir: str, 
        no_seeds: int
        ) -> List[str]:
        vol_dir = "vol_exp_%02d" % vol_exp
        no_ip3r_dir = "ip3r_%05d" % no_ip3r
        ddir = os.path.join(data_dir, vol_dir, no_ip3r_dir, ip3_dir)

        # Construct fnames
        fnames = []
        for seed in range(0,no_seeds):
            fname = os.path.join(ddir,"%04d.txt" % seed)
            fnames.append(fname)

        return fnames

    @staticmethod
    def import_gillespie_ssa_from_data_desc(
        data_desc: DataDesc, 
        data_dir: str, 
        vol_exp: int, 
        no_ip3r: int, 
        ip3_dir: str
        ) -> np.array:

        fnames = ImportHelper.create_fnames(data_dir,vol_exp,no_ip3r,ip3_dir,data_desc.no_seeds)

        ret = np.zeros(shape=(len(data_desc.times),len(fnames),len(data_desc.species)))
        for i,time in enumerate(data_desc.times):
            ret[i] = ImportHelper.import_gillespie_ssa(fnames, time, data_desc.species)
        
        return ret

    @staticmethod
    def import_gillespie_ssa_from_data_desc_at_tpt(
        data_desc: DataDesc, 
        data_dir: str, 
        vol_exp: int, 
        no_ip3r: int, 
        ip3_dir: str,
        time: int
        ) -> np.array:

        fnames = ImportHelper.create_fnames(data_dir,vol_exp,no_ip3r,ip3_dir,data_desc.no_seeds)
        return ImportHelper.import_gillespie_ssa(fnames, time, data_desc.species)

    @staticmethod
    def import_gillespie_ssa(fnames: List[str], time: float, species: List[str]) -> np.array:
        # Read first fname
        ff = pd.read_csv(fnames[0], sep=" ")

        # Find row
        times = ff['t'].to_numpy()
        idxs = np.where(abs(times - time) < 1e-8)[0]
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
