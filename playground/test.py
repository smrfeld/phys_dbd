from physDBD import ImportHelper, DataDesc, ParamsTraj, RxnSpec, RxnInputsLayer, ParamsTETraj, RxnModel

import numpy as np

import tensorflow as tf

import sys

data_desc = DataDesc(
    no_seeds=50,
    time_start=0,
    time_end=20,
    time_interval=0.5,
    species=["ca2i","ip3"]
)

data = ImportHelper.import_gillespie_ssa_from_data_desc(
    data_desc=data_desc,
    data_dir="/Users/oernst/Documents/papers/2021_03_draft/stochastic_simulations/ml_training_data/data_gillespie/",
    vol_exp=12,
    no_ip3r=100,
    ip3_dir="ip3_0p100"
)

if False:

    # Create params traj and export
    muh = np.zeros(1)
    varh_diag = np.ones(1)
    params_traj = ParamsTraj.fromPCA(data, data_desc.times, muh, varh_diag)

    # Export
    params_traj.export("cache_params.txt")

else:

    # Import params traj
    params_traj = ParamsTraj.fromFile("cache_params.txt", nv=2, nh=1)

if False:

    # Differentiate
    paramsTE_traj = params_traj.differentiate_with_TVR(alpha=1.0, no_opt_steps=10)

    # Export
    paramsTE_traj.export("cache_derivs.txt")

else:

    # Import paramsTE_traj
    paramsTE_traj = ParamsTETraj.fromFile("cache_derivs.txt", nv=2, nh=1)

train_inputs = params_traj.get_tf_inputs_assuming_params0()
train_outputs, \
    train_outputs_mean, \
        train_outputs_std = paramsTE_traj.get_tf_outputs_normalized_assuming_params0(0.2)

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
rxn_layer = RxnInputsLayer(
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
    tf.keras.layers.Dense(128, activation='relu')
    # tf.keras.layers.Dropout(0.2)
])

class MyModel(RxnModel):
    def call(self, input_tensor, training=False):
        out = super().call(input_tensor, training=training)
        return out

model = MyModel(nv,nh,rxn_layer,subnet)

# Normalize
model.calculate_rxn_normalization(train_inputs)

# Build the model by calling it on real data
input_build = params_traj.params_traj[0].get_tf_input_assuming_params0(tpt=1)
print("Test input: ", input_build)
output_build = model(input_build)
print("Test output: ", output_build)

# From what @AniketBote wrote, if you compile your model with the run_eagerly=True flag 
# then you should see the values of x, y in your train_step, 
# ie  model.compile(optimizer, loss, run_eagerly=True).
loss_fn = tf.keras.losses.MeanSquaredError()
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt,
              loss=loss_fn,
              metrics=['accuracy'],
              run_eagerly=False)

model.fit(train_inputs, train_outputs, epochs=5)