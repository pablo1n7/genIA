import tqdm
import numpy as np

simu = 'data'
print(simu)
stack_exp = []
for i in tqdm.tqdm(range(1,60)):#/home/pablo1n7/orlando_umap/clean/newsimu/simu_1/rand0
    shap_values_rand = np.load(f'{simu}/rand{i}/nfi_values.npy')
    mean_abs_rand = np.abs(shap_values_rand).mean(axis=0)
    stack_exp.append(mean_abs_rand)

null_distribution = np.vstack(stack_exp)

np.save(f'{simu}/null_distribution.npy', null_distribution)