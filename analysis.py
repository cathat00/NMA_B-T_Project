# Imports for this script
import os
import numpy as np
import modules.configs as configs
from os.path import join
from tqdm import tqdm
from modules.plots import MultiSeedSummaryViz

root = "C:/Users/caleb/OneDrive/Desktop/results"
n_seeds, n_experiments, n_evals = 5, 20, 800
all_n_over_thresh = np.zeros((n_seeds, n_experiments))
all_prs = np.zeros((n_seeds, n_experiments))

for seed_folder in tqdm(os.listdir(root)):

    results_folders = os.listdir(join(root, seed_folder))

    for folder in results_folders:
        
        if os.path.isdir(join(root, seed_folder, folder)):

            datafile = join(root, seed_folder, folder, "result.pkl")
            result = configs.load_data(datafile)

            seed_idx = result['params']['random_seed'] - 1
            exp_idx = result['task'].ntargets - 1 # Experiment index

            # -- Store participation ratio
            all_prs[seed_idx, exp_idx] = result['manifold_data']['pr']

            # -- Compute and store the # of PCs over the 90% threshold
            evals = result['manifold_data']['ev']
            var_explained = evals / np.sum(evals) # Var. explained by each PC
            cumvar = np.cumsum(var_explained, axis=-1) # Cumulative var. explained by PCs
            n_over_thresh = np.count_nonzero(cumvar <= 0.9) # Number of PCs required for cumvar >= 0.9
            all_n_over_thresh[seed_idx, exp_idx] = n_over_thresh # Store in matrix for plotting

viz = MultiSeedSummaryViz(np.arange(1,21), all_prs, all_n_over_thresh)
viz.save("./test.html", overwrite=True)