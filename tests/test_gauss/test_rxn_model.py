from dataclasses import dataclass
from physDBD import Params0Gauss, ImportHelper, Params0GaussTraj, \
    DParams0GaussTraj, RxnGaussModel, RxnInputsGaussLayer
import numpy as np
import os
import shutil
import tensorflow as tf

class TestRxnGaussModel:

    fnames = [
        "../data_test/0000.txt",
        "../data_test/0001.txt",
        "../data_test/0002.txt",
        "../data_test/0003.txt",
        "../data_test/0004.txt"
        ]
    species = ["ca2i","ip3"]

    def import_params(self, time: float) -> Params0Gauss:
        data = ImportHelper.import_gillespie_ssa_at_time(
            fnames=self.fnames,
            time=time,
            species=self.species
        )

        params = Params0Gauss.fromData(data)
        return params

    def create_params_traj(self) -> Params0GaussTraj:
        return Params0GaussTraj(
            times=np.array([0.2,0.3,0.4,0.5,0.6,0.7]),
            params0_traj=[
                self.import_params(0.2),
                self.import_params(0.3),
                self.import_params(0.4),
                self.import_params(0.5),
                self.import_params(0.6),
                self.import_params(0.7)
                ]
            )
    
    def create_params_te_traj(self) -> DParams0GaussTraj:
        pt = self.create_params_traj()

        ptTE = pt.differentiate_with_TVR(
            alphas={"wt00": 1.0},
            no_opt_steps=10,
            non_zero_vals=["wt00"]
        )

        return ptTE

    def test_rxn_normalization(self):

        # Freqs, coffs for fourier
        nv = 2
        nh = 1
        freqs = np.random.rand(3)
        muh_sin_coeffs_init = np.random.rand(3)
        muh_cos_coeffs_init = np.random.rand(3)
        cholhv_sin_coeffs_init = np.random.rand(3)
        cholhv_cos_coeffs_init = np.random.rand(3)
        cholh_sin_coeffs_init = np.random.rand(3)
        cholh_cos_coeffs_init = np.random.rand(3)
        non_zero_idx_pairs_vv = [(0,0),(1,0),(1,1)]
        non_zero_idx_pairs_hv = [(0,0),(0,1)]
        non_zero_idx_pairs_hh = [(0,0)]

        # Rxns
        rxn_specs = [
            ("BIRTH",0),
            ("DEATH",1),
            ("EAT",2,1)
            ]

        # Reaction input layer
        rxn_lyr = RxnInputsGaussLayer.construct(
            nv=nv,
            nh=nh,
            freqs=freqs,
            muh_sin_coeffs_init=muh_sin_coeffs_init,
            muh_cos_coeffs_init=muh_cos_coeffs_init,
            cholhv_sin_coeffs_init=cholhv_sin_coeffs_init,
            cholhv_cos_coeffs_init=cholhv_cos_coeffs_init,
            cholh_sin_coeffs_init=cholh_sin_coeffs_init,
            cholh_cos_coeffs_init=cholh_cos_coeffs_init,
            non_zero_idx_pairs_vv=non_zero_idx_pairs_vv,
            non_zero_idx_pairs_hv=non_zero_idx_pairs_hv,
            non_zero_idx_pairs_hh=non_zero_idx_pairs_hh,
            rxn_specs=rxn_specs
            )

        subnet = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2)
        ])

        rxn_model = RxnGaussModel.construct(nv,nh,rxn_lyr,subnet)

        # Calculate normalizations
        pt = self.create_params_traj()
        inputs = pt.get_tf_inputs(
            tpt_start_inc=0,
            tpt_end_exc=pt.nt-1,
            non_zero_idx_pairs_vv=non_zero_idx_pairs_vv
            )
        # rxn_model.calculate_rxn_normalization(inputs, percent=0.8)

        # Compile
        # Must compile first in order for DenseLayer output to be saved correctly
        rxn_model.compile(
            optimizer='adam',
            loss=tf.losses.MeanSquaredError()
            )

        # Call the model once to build it
        outputs = rxn_model(inputs)
        print("Outputs without norm: ", outputs)

        rxn_model.calculate_rxn_normalization(rxn_lyr, inputs, percent=0.8)

        print("Rxn mean: ", rxn_model.rxn_mean)
        print("Rxn std dev: ", rxn_model.rxn_std_dev)

        outputs_1 = rxn_model(inputs)
        print("Outputs with norm: ", outputs_1)

        # Test save; call the model once to build it first
        rxn_model.save("saved_models/rxn_model", save_traces=False)

        # Test load
        print(rxn_model)
        rxn_model_loaded = tf.keras.models.load_model("saved_models/rxn_model", custom_objects={"RxnGaussModel": RxnGaussModel})
        print(rxn_model)

        # Check type
        assert type(rxn_model) is type(rxn_model_loaded)

        # Run again
        outputs_2 = rxn_model_loaded(inputs)
        print("Outputs after loading: ", outputs_2)

        # Check outputs are same after loading
        for key in outputs_1.keys():
            diff = abs(outputs_2[key].numpy() - outputs_1[key].numpy())
            max_diff = np.max(diff)
            print("max_diff: ", key, max_diff)
            assert max_diff < 1.e-10
        
        # Remove
        if os.path.isdir("saved_models"):
            shutil.rmtree("saved_models")