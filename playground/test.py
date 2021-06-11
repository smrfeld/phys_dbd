from physPCA import ImportHelper, DataDesc, ParamsTraj

import numpy as np

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

muh = np.zeros(1)
varh_diag = np.ones(1)
params_traj = ParamsTraj.fromPCA(data, data_desc.times, muh, varh_diag)

# Export
params_traj.export("cache_params.txt")

# Differentiate
paramsTE_traj = params_traj.differentiate_with_TVR(alpha=1.0, no_opt_steps=10)

# Export
paramsTE_traj.export("cache_derivs.txt")