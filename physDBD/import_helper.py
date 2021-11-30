from .data_desc import DataDesc

import pandas as pd
import numpy as np
from typing import List

import os

class ImportHelper:

    @staticmethod
    def create_fnames(
        data_dir: str,
        seed_start_inc: int,
        seed_end_exc: int,
        zero_padding: int
        ) -> List[str]:
        """Create fnames for different seeds with zero padding 4

        Args:
            data_dir (str): Data directory
            seed_start_inc (int): Start seed index inclusive
            seed_end_exc (int): End seed index exclusive
            zero_padding (int): Zero padding for fnames

        Returns:
            List[str]: List of filenames
        """

        # Construct fnames
        fstr = "%0" + str(zero_padding) + "d.txt"
        fnames = []
        for seed in range(seed_start_inc,seed_end_exc):
            fname = os.path.join(data_dir,fstr % seed)
            fnames.append(fname)

        return fnames

    @staticmethod
    def import_gillespie_ssa_from_data_desc(
        data_desc: DataDesc, 
        data_dir: str,
        zero_padding: int
        ) -> np.array:
        """Import Gillespie SSA data

        Args:
            data_desc (DataDesc): Data description
            data_dir (str): Data directory
            zero_padding (int): Zero padding for fnames

        Returns:
            np.array: Numpy array of size (no_times, no_seeds, no_species)
        """

        fnames = ImportHelper.create_fnames(
            data_dir=data_dir,
            seed_start_inc=data_desc.seed_start_inc,
            seed_end_exc=data_desc.seed_end_exc,
            zero_padding=zero_padding
            )

        ret = ImportHelper.import_gillespie_ssa_whole_file(fnames, data_desc.times, data_desc.species)
        ret = np.transpose(ret, axes=[1,0,2])

        return ret

    @staticmethod
    def import_gillespie_ssa_from_data_desc_at_time(
        data_desc: DataDesc, 
        data_dir: str, 
        time: float
        ) -> np.array:
        """Import Gillespie SSA data at a single timepoint

        Args:
            data_desc (DataDesc): Data description
            data_dir (str): Data directory
            time (float): Time (real time)

        Returns:
            np.array: Numpy array of size (no_seeds, no_species)
        """

        fnames = ImportHelper.create_fnames(
            data_dir=data_dir,
            seed_start_inc=data_desc.seed_start_inc,
            seed_end_exc=data_desc.seed_end_exc
            )
        return ImportHelper.import_gillespie_ssa_at_time(fnames, time, data_desc.species)

    @staticmethod
    def import_gillespie_ssa_at_time(
        fnames: List[str], 
        time: float, 
        species: List[str]
        ) -> np.array:
        """Import seeds from different files at a single time

        Args:
            fnames (List[str]): The different files corresponding to the seeds
            time (float): Time (real time)
            species (List[str]): Species

        Raises:
            ValueError: If species not found

        Returns:
            np.array: Array of size (no_seeds, no_species)
        """

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

    @staticmethod
    def import_gillespie_ssa_whole_file(
        fnames: List[str], 
        times: List[float], 
        species: List[str]
        ) -> np.array:
        """Import Gillespie SSA whole file and extract multiple times for different seeds

        Args:
            fnames (List[str]): List of files corresponding to different seeds
            times (List[float]): List of times (real time) to extract
            species (List[str]): Species

        Raises:
            ValueError: If species not found

        Returns:
            np.array: array of size (no_times, no_seeds, no_species)
        """

        # Read first fname
        ff = pd.read_csv(fnames[0], sep=" ")

        # Find rows for times
        f_times = ff['t'].to_numpy()
        time_idxs = []
        for time in times:
            idxs = np.where(abs(f_times - time) < 1e-8)[0]
            if len(idxs) != 1:
                raise ValueError("Could not find time: %f in the data" % time)
            
            # Add 1 for the header
            idx = idxs[0]
            time_idxs.append(idx)

        # Data to return
        ret = np.zeros(shape=(len(fnames),len(times),len(species)))

        # Import
        for i,fname in enumerate(fnames):
            ff = pd.read_csv(fname, header=0, sep=" ")
            ff = ff.iloc[time_idxs]
            ret[i] = ff[species].to_numpy()
        
        return ret