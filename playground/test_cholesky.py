from physDBD import ParamsGauss, DataDesc, ImportHelper

import numpy as np
import os

data_desc = DataDesc(
    no_seeds=50,
    time_start=0,
    time_end=20,
    time_interval=0.5,
    species=["ca2i","ip3"]
)

data_dir = "playground_data"
vol_exp = 12
no_ip3r = 100
ip3_dir = "ip3_0p100"
vol_dir = "vol_exp_%02d" % vol_exp
no_ip3r_dir = "ip3r_%05d" % no_ip3r
data_dir = os.path.join(data_dir, vol_dir, no_ip3r_dir, ip3_dir)

data = ImportHelper.import_gillespie_ssa_from_data_desc(
    data_desc=data_desc,
    data_dir=data_dir
    )

print(data.shape)

# Params
nh = 4
nv = data.shape[2]
params = ParamsGauss.fromDataStd(data[1],nh)
print(params)
print(params.cov)

mu_h_new = np.random.rand(nh)
chol_vh_new = np.random.rand(nh,nv)
chol_h_new = np.tril(np.random.rand(nh,nh))
params.convert_latent_space(
    mu_h_new=mu_h_new,
    chol_vh_new=chol_vh_new,
    chol_h_new=chol_h_new
    )
print(params)
print(params.cov)