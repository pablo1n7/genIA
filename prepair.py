# %%
import pandas as pd

# %%
import numpy as np

# %%
caso = 'InterG_11'

# %%
geneticos = pd.read_csv(f'{caso}/genotype_sim.raw', sep='\t')

# %%
geneticos.columns

# %%
gen = np.array(geneticos[list(geneticos.columns)[6:]])

# %%
feno = pd.read_csv(f'{caso}/fenodata.txt', sep=' ')['bmi']

# %%
np.save(f'{caso}/x.npy', gen)
np.save(f'{caso}/iids_all.npy', np.array(geneticos['IID']))
np.save(f'{caso}/feno_all.npy', np.array(feno))
np.save(f'{caso}/feno_all_c.npy', np.array(feno))
np.save(f'{caso}/columns.npy', np.array(list(geneticos.columns)[6:]))

# %%



