import pandas as pd
import numpy as np
#simu = '/media/disk/InterG/InterG_10'
simu = 'data'
shap_values = np.load(f'{simu}/rand0/nfi_values.npy')
null_distribution = np.load(f'{simu}/null_distribution.npy')
mean_abs_shap_real = np.abs(shap_values).mean(axis=0)

n_permutaciones = null_distribution.shape[0]
p_values = (np.sum(null_distribution >= mean_abs_shap_real, axis=0) + 1) / (n_permutaciones + 1)
np.save(f'{simu}/p_values_emp.npy', p_values)