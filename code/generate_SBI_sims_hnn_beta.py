import os
import dill
import joblib
from utils import (linear_scale_forward, log_scale_forward, start_cluster,
                   run_hnn_sim, hnn_beta_param_function, UniformPrior,
                   load_prerun_simulations, filter_borders, filter_nzeros, filter_peakproeminence,
                   filter_removeoutliers, filter_peaktime, PriorBetaFiltered)
from hnn_core import jones_2009_model
import numpy as np
from tqdm import tqdm
from torch import optim

nsbi_sims = 100_000
num_prior_fits = 2
tstop = 500
dt = 0.5

net = jones_2009_model()

save_path = '../data/'

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

start_cluster() # reserve resources for HNN simulations

# Define filtering steps
filters = [filter_borders, filter_nzeros, filter_peakproeminence, filter_peaktime, filter_removeoutliers]
#filters = [filter_borders]

for flow_idx in range(num_prior_fits):
    if flow_idx == 0:
        prior = UniformPrior(parameters=list(prior_dict.keys()))
    else:
        prior = prior_filtered
        
    theta_samples = prior.sample((nsbi_sims,))
    
    save_suffix = f'sbi_{flow_idx}'
    run_hnn_sim(net=net, param_function=hnn_beta_param_function, prior_dict=prior_dict,
                theta_samples=theta_samples, tstop=tstop, save_path=save_path, save_suffix=save_suffix)
    
    dpl_filter = np.load(f'{save_path}/sbi_sims/dpl_sbi_{flow_idx}.npy')
    theta_filter = np.load(f'{save_path}/sbi_sims/theta_sbi_{flow_idx}.npy')
    
    for filter_func in filters:
        dpl_filter, theta_filter = filter_func(dpl_filter, theta_filter)
    
    prior_filtered = PriorBetaFiltered(parameters=list(prior_dict.keys()))
    optimizer = optim.Adam(prior_filtered.flow.parameters())

    num_iter = 5000
    for i in tqdm(range(num_iter)):
        optimizer.zero_grad()
        loss = -prior_filtered.flow.log_prob(inputs=theta_filter).mean()
        loss.backward()
        optimizer.step()
    state_dict = prior_filtered.flow.state_dict()
    joblib.dump(state_dict, f'{save_path}/flows/prior_filtered_flow_{flow_idx}.pkl')



#os.system('scancel -u ntolley')