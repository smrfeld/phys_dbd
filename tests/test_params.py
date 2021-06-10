from physPCA import Params, ImportHelper
import numpy as np

class TestParams:

    fnames = [
        "data_test/0000.txt",
        "data_test/0001.txt",
        "data_test/0002.txt",
        "data_test/0003.txt",
        "data_test/0004.txt"
        ]
    time = 0.4
    species = ["ca2i","ip3"]

    def test_params(self):

        data = ImportHelper.import_gillespie_ssa(
            fnames=self.fnames,
            time=self.time,
            species=self.species
        )

        muh = np.zeros(1)
        varh_diag = np.ones(1)
        params = Params.fromPCA(data,muh,varh_diag)
        print(params)

TestParams().test_params()