import os
import dill
from utils import (linear_scale_forward, log_scale_forward, start_cluster,
                   run_hnn_sim, hnn_beta_param_function, UniformPrior)
from hnn_core import jones_2009_model

nsbi_sims = 100_000
tstop = 
dt = 0.5

net = jones_2009_model()

save_path = '../data/'
save_suffix = 'sbi'

prior_dict = {'gbar_evprox_1_L2Pyr_ampa': {'bounds': (1e-10, 1e-1), 'rescale_function': log_scale_forward}, 
               'gbar_evprox_1_L5Pyr_ampa': {'bounds': (1e-10, 1e-1), 'rescale_function': log_scale_forward}, 
               'gbar_evdist_1_L2Pyr_ampa': {'bounds': (1e-10, 1e-1), 'rescale_function': log_scale_forward}, 
               'gbar_evdist_1_L5Pyr_ampa': {'bounds': (1e-10, 1e-1), 'rescale_function': log_scale_forward},
               'sigma_t_evprox_1': {'bounds': (1, 100), 'rescale_function': linear_scale_forward},
               'sigma_t_evdist_1': {'bounds': (1, 100), 'rescale_function': linear_scale_forward},
               't_evprox_1': {'bounds': (200, 300), 'rescale_function': linear_scale_forward},
               't_evdist_1': {'bounds': (200, 300), 'rescale_function': linear_scale_forward}} 




with open(f'{save_path}/sbi_sims/prior_dict.pkl', 'wb') as f:
    dill.dump(prior_dict, f)

sim_metadata = {'nsbi_sims': nsbi_sims, 'tstop': tstop, 'dt': dt, 'gid_ranges': net.gid_ranges}
with open(f'{save_path}/sbi_sims/sim_metadata.pkl', 'wb') as f:
    dill.dump(sim_metadata, f)

prior = UniformPrior(parameters=list(prior_dict.keys()))
theta_samples = prior.sample((nsbi_sims,))

start_cluster() # reserve resources for HNN simulations

run_hnn_sim(net=net, param_function=hnn_beta_param_function, prior_dict=prior_dict,
            theta_samples=theta_samples, tstop=tstop, save_path=save_path, save_suffix=save_suffix)

#os.system('scancel -u ntolley')