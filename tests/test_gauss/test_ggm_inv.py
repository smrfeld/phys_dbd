from helpers_test import SingleLayerModel
from physDBD import GGMmultPrecCovLayer, invert_ggm, invert_ggm_chol, construct_mat, invert_ggm_bmlike, GGMInvPrecToCovMatLayer, GGMInvModelBMLike, construct_mat_non_zero

import numpy as np
import tensorflow as tf

class TestGGMInv:

    def assert_equal_dicts(self, x_out, x_out_true):
        for key, val_true in x_out_true.items():
            val = x_out[key]

            self.assert_equal_arrs(val,val_true)

    def assert_equal_arrs(self, x_out, x_out_true):
        tol = 5.e-4
        assert np.max(abs(x_out-x_out_true)) < tol

    def save_load_model(self, lyr, x_in):

        # Test save; call the model once to build it first
        model = SingleLayerModel(lyr)
        x_out = model(x_in)

        print(model)
        model.save("saved_models/model", save_traces=False)

        # Test load
        model_rel = tf.keras.models.load_model("saved_models/model")
        print(model_rel)

        # Check types match!
        # Otherwise we may have: tensorflow.python.keras.saving.saved_model.load.XYZ instead of XYZ
        assert type(model_rel) is type(model)

    def test_inv_prec_to_cov_mat(self):

        n = 2
        non_zero_idx_pairs = [(0,0),(1,0),(1,1)]

        # Create layer
        lyr = GGMmultPrecCovLayer.construct(
            n=n,
            non_zero_idx_pairs=non_zero_idx_pairs,
            init_diag_val=10.0
        )

        # Output
        x_in = {"cov_mat": np.array([[0.1,0.0],[0.0,0.1]])}
        x_out = lyr(x_in)

        print("Outputs: ",x_out)

        x_out_true = np.eye(n)

        self.assert_equal_arrs(x_out, x_out_true)

        self.save_load_model(lyr, x_in)

    def test_invert_ggm_normal_n2(self):

        n = 2
        non_zero_idx_pairs = [(0,0),(1,0),(1,1)]
        cov_mat = np.array([[0.1,0.0],[0.0,0.1]])

        prec_mat_non_zero, final_loss = invert_ggm(
            n=n,
            non_zero_idx_pairs=non_zero_idx_pairs,
            cov_mat=cov_mat
        )

        print(final_loss)
        assert final_loss < 1.e-10

        prec_mat = construct_mat(n,non_zero_idx_pairs,prec_mat_non_zero)
        identity_check = np.dot(prec_mat,cov_mat)

        print(prec_mat)
        print(identity_check)

        self.assert_equal_arrs(identity_check, np.eye(n))

    def test_invert_ggm_bmlike_n2(self):

        n = 3
        non_zero_idx_pairs = [(0,0),(1,0),(1,1),(2,1),(2,2)]
        cov_mat = np.array([
            [10.0,5.0,2.0],
            [5.0,20.0,4.0],
            [2.0,4.0,30.0]
            ])
        
        # Invert 
        prec_mat_init = np.linalg.inv(cov_mat)
        prec_mat_non_zero_init = construct_mat_non_zero(n,non_zero_idx_pairs,prec_mat_init)
        print("Init prec mat: ", prec_mat_non_zero_init)

        lyr = GGMInvPrecToCovMatLayer(
            n=n,
            non_zero_idx_pairs=non_zero_idx_pairs,
            non_zero_vals=prec_mat_non_zero_init
            )

        model = GGMInvModelBMLike(inv_lyr=lyr)

        print("Trying:")
        batch_size = 2
        inputs = np.random.rand(batch_size,1)
        print(model(inputs))
        print("Should be")
        print(construct_mat_non_zero(n,non_zero_idx_pairs,cov_mat))

        loss_fn = tf.keras.losses.MeanSquaredError()
        opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        model.compile(optimizer=opt,
                    loss=loss_fn,
                    run_eagerly=False)

        max_batch_size = 100
        inputs = np.full(
            shape=(max_batch_size,1),
            fill_value=np.array([2])
            )
        
        cov_mat_non_zero = construct_mat_non_zero(n,non_zero_idx_pairs,cov_mat)
        outputs = np.full(
            shape=(max_batch_size,len(cov_mat_non_zero)),
            fill_value=cov_mat_non_zero
            )

        # Train!
        model.fit(
            inputs, 
            outputs, 
            epochs=100, 
            batch_size=2
            )

        prec_mat_non_zero = model.inv_lyr.non_zero_vals.numpy()

        prec_mat = construct_mat(n,non_zero_idx_pairs,prec_mat_non_zero)
        print(prec_mat)
        print(np.linalg.inv(prec_mat))
        print("Should equal")
        print(cov_mat)

        '''
        prec_mat_non_zero = invert_ggm_bmlike(
            n=n,
            non_zero_idx_pairs=non_zero_idx_pairs,
            cov_mat=cov_mat,
            epochs=4000,
            learning_rate=0.1
        )

        prec_mat = construct_mat(n,non_zero_idx_pairs,prec_mat_non_zero)
        print(prec_mat)
        print(np.linalg.inv(prec_mat))
        print("Should equal")
        print(cov_mat)
        '''
    
    def test_invert_ggm_bmlike_n3(self):

        n = 3
        non_zero_idx_pairs = [(0,0),(1,0),(1,1),(2,1),(2,2)]
        cov_mat_non_zero = np.array([10.0,5.0,20.0,4.0,30.0])

        prec_mat_non_zero, model = invert_ggm_bmlike(
            n=n,
            non_zero_idx_pairs=non_zero_idx_pairs,
            cov_mat_non_zero=cov_mat_non_zero,
            epochs=100,
            learning_rate=0.001
        )

        prec_mat = construct_mat(n,non_zero_idx_pairs,prec_mat_non_zero)
        print("Precision mat found:", prec_mat)

        cov_mat_found = np.linalg.inv(prec_mat)
        print("Cov mat found:", cov_mat_found)

        cov_mat_non_zero_found = construct_mat_non_zero(n,non_zero_idx_pairs,cov_mat_found)
        print("Non-zero elements found:", cov_mat_non_zero_found)
        print("Should equal given:")
        print(cov_mat_non_zero)

    def test_invert_ggm_bmlike_n10(self):

        n = 10
        
        # Non zero elements
        non_zero_idx_pairs = []
        # All diagonal (required)
        for i in range(0,n):
            non_zero_idx_pairs.append((i,i))
        # Some off diagonal < n choose 2
        max_no_off_diagonal = int((n-1)*n/2)
        no_off_diagonal = np.random.randint(low=0,high=max_no_off_diagonal)
        print("No non-zero off-diagonal elements:",no_off_diagonal,"max possible:",max_no_off_diagonal)
        idx = 0
        while idx < no_off_diagonal:
            i = np.random.randint(low=1,high=n)
            j = np.random.randint(low=0,high=i)
            if not (i,j) in non_zero_idx_pairs:
                non_zero_idx_pairs.append((i,j))
                idx += 1

        print("Non-zero elements:",non_zero_idx_pairs)

        # Random cov mat using chol decomposition
        # Diagonal = positive => unique
        chol = np.tril(np.random.rand(n,n))
        cov_mat = np.dot(chol,np.transpose(chol))
        
        print("Cov mat:",cov_mat)

        cov_mat_non_zero = construct_mat_non_zero(n,non_zero_idx_pairs,cov_mat)
        
        print("Non-zero constraints:",cov_mat_non_zero)

        result = invert_ggm_bmlike(
            n=n,
            non_zero_idx_pairs=non_zero_idx_pairs,
            cov_mat_non_zero=cov_mat_non_zero,
            epochs=100,
            learning_rate=0.01
        )

        result.report()

    def test_invert_ggm_normal_n3(self):

        n = 3
        non_zero_idx_pairs = [(0,0),(1,0),(1,1),(2,1),(2,2)]
        cov_mat = np.array([
            [10.0,5.0,2.0],
            [5.0,20.0,4.0],
            [2.0,4.0,30.0]
            ])

        prec_mat_non_zero, final_loss = invert_ggm(
            n=n,
            non_zero_idx_pairs=non_zero_idx_pairs,
            cov_mat=cov_mat,
            epochs=5000,
            learning_rate=0.001
        )

        print(final_loss)
        # assert final_loss < 1.e-10

        prec_mat = construct_mat(n,non_zero_idx_pairs,prec_mat_non_zero)
        print(prec_mat)
        print(np.linalg.inv(prec_mat))
        print("Should be")
        print(cov_mat)
        identity_check = np.dot(prec_mat,cov_mat)

        print(identity_check)

        self.assert_equal_arrs(identity_check, np.eye(n))

    def test_invert_ggm_chol(self):

        n = 2
        non_zero_idx_pairs = [(0,0),(1,0),(1,1)]
        cov_mat = np.array([[0.1,0.0],[0.0,0.1]])

        chol, final_loss = invert_ggm_chol(
            n=n,
            non_zero_idx_pairs=non_zero_idx_pairs,
            cov_mat=cov_mat
        )

        print(final_loss)
        print(chol)

        assert final_loss < 1.e-10
        # self.assert_equal_arrs(prec_mat, np.array([10.0,0.0,10.0]))