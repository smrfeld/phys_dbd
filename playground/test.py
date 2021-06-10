from physPCA import ImportHelper, DataDesc, Params, export_params_traj

import numpy as np

data_desc = DataDesc(
    no_seeds=50,
    time_start=0,
    time_end=20,
    time_interval=0.5,
    species=["ca2i","ip3"]
)

params_traj = []
for time in data_desc.times:
    data = ImportHelper.import_gillespie_ssa_from_data_desc_at_tpt(
        data_desc=data_desc,
        data_dir="/Users/oernst/Documents/papers/2021_03_draft/stochastic_simulations/ml_training_data/data_gillespie/",
        vol_exp=12,
        no_ip3r=100,
        ip3_dir="ip3_0p100",
        time=time
    )

    muh = np.zeros(1)
    varh_diag = np.ones(1)
    params = Params.fromPCA(data, muh, varh_diag)
    params_traj.append(params)

# Export
export_params_traj("cache.txt",data_desc.times,params_traj)