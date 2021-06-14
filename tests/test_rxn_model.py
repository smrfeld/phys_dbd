from dataclasses import dataclass
from physDBD import Params, ImportHelper, ParamsTraj, ParamsTETraj, RxnModel, RxnSpec, RxnInputsLayer
import numpy as np
import os
import tensorflow as tf

class TestRxnModel:

    fnames = [
        "data_test/0000.txt",
        "data_test/0001.txt",
        "data_test/0002.txt",
        "data_test/0003.txt",
        "data_test/0004.txt"
        ]
    species = ["ca2i","ip3"]

    def import_params(self, time: float) -> Params:
        data = ImportHelper.import_gillespie_ssa(
            fnames=self.fnames,
            time=time,
            species=self.species
        )

        muh = np.zeros(1)
        varh_diag = np.ones(1)
        params = Params.fromPCA(data,muh,varh_diag)
        return params

    def create_params_traj(self) -> ParamsTraj:
        return ParamsTraj(
            times=np.array([0.2,0.3,0.4,0.5,0.6,0.7]),
            params_traj=[
                self.import_params(0.2),
                self.import_params(0.3),
                self.import_params(0.4),
                self.import_params(0.5),
                self.import_params(0.6),
                self.import_params(0.7)
                ]
            )
    
    def create_params_te_traj(self) -> ParamsTETraj:
        pt = self.create_params_traj()

        ptTE = pt.differentiate_with_TVR(
            alpha=1.0,
            no_opt_steps=10
        )

        return ptTE

    def test_rxn_normalization(self):

        # Freqs, coffs for fourier
        nv = 2
        nh = 1
        freqs = np.random.rand(3)
        muh_sin_coeffs_init = np.random.rand(3)
        muh_cos_coeffs_init = np.random.rand(3)
        varh_sin_coeffs_init = np.random.rand(3)
        varh_cos_coeffs_init = np.random.rand(3)

        # Rxns
        rxn_specs = [
            (RxnSpec.BIRTH,0),
            (RxnSpec.DEATH,1),
            (RxnSpec.EAT,2,1)
            ]

        # Reaction input layer
        rxn_lyr = RxnInputsLayer(
            nv=nv,
            nh=nh,
            freqs=freqs,
            muh_sin_coeffs_init=muh_sin_coeffs_init,
            muh_cos_coeffs_init=muh_cos_coeffs_init,
            varh_sin_coeffs_init=varh_sin_coeffs_init,
            varh_cos_coeffs_init=varh_cos_coeffs_init,
            rxn_specs=rxn_specs
            )

        subnet = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2)
        ])

        rxn_model = RxnModel(nv,nh,rxn_lyr,subnet)

        pt = self.create_params_traj()

        # Inputs/outputs
        inputs = pt.get_tf_inputs_assuming_params0()

        outputs = rxn_model(inputs)
        print("Outputs without norm: ", outputs)

        rxn_model.calculate_rxn_normalizations(inputs)

        print("Rxn mean: ", rxn_model.rxn_mean)
        print("Rxn std dev: ", rxn_model.rxn_std_dev)

        outputs = rxn_model(inputs)
        print("Outputs without norm: ", outputs)