# Imports for this script
import numpy as np
import modules.configs as configs
import matplotlib.pyplot as plt
from os.path import join
from modules.plots import TaskViz, ManifoldViz, ExperimentSummaryViz
import sklearn.linear_model as lm


# ====================
# == Initialization ==
# ====================

seeds = [1,2,3,4,5]
experiments = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
eigenval_list = []


# =====================
# == Run Experiments ==   
# =====================

for seed in seeds:

  root_out_dir = f"./results/seed{seed}"

  for ntargets in experiments:


    # ======================
    # == Setup Experiment ==
    # ======================

    exp_name = f"{ntargets}Targets" # Name of this experiment
    exp_out_dir = join(root_out_dir, exp_name)

    cfg = configs.BasicExperimentConfig(ntrials=5, ntargets=ntargets, seed=seed)
    task = cfg.task # The task the RNN will learn
    rnn = cfg.rnn # Recurrent Neural Network
    bci = cfg.bci # Brain computer interface

    print(f"Starting experiment: {exp_name}")
    print(f"-- Random Seed: {seed}")
    print(f"-- Stimulus Shape: {task.stimuli.shape}")
    print(f"-- Targets Shape: {task.targets.shape}")


    # =======================
    # == Simulate Learning ==
    # =======================

    # -- Train the RNN
    rnn.relearn(cfg.ntrials, task.stimuli,
                task.stim_length, bci.decoder,
                cfg.feedback, task.targets)

    # -- Compute manifold
    manifold_out = rnn.calculate_manifold(
       task.stimuli, 
       task.stim_length, 
       cfg.ntrials_manifold
    )

    proj = manifold_out['xi2'] # Reshaped projection
    activity = manifold_out['activity_reshaped']
    order = manifold_out['order'] # Target indices for each trial
    evecs = manifold_out['evec'] # Eigenvectors
    eigenval_list.append(manifold_out['ev'])

    # -- Train BCI
    bci.train(proj,task.targets[:,task.stim_length:,:],evecs,order)

    # ===========================================
    # == Visualize Results For This Experiment ==
    # ===========================================
    
    manifold_viz = ManifoldViz(manifold_out)
    task_viz = TaskViz(task, activity, bci, order, dt=rnn.dt)
    manifold_viz.save(join(exp_out_dir, "Manifold.html"), overwrite=True)
    task_viz.save(join(exp_out_dir, "Task.html"), overwrite=True)


    # ======================================
    # == Save Results For This Experiment ==
    # ======================================
    print("Saving results for experiment...")
    cfg.save(join(exp_out_dir, "result.pkl"), manifold_out, overwrite=True)
    print("Experiment complete!\n")


  # ==============================================
  # == Save Summary Results For All Experiments ==
  # ==============================================
  print("Saving summary results for all experiments...")
  summary_viz = ExperimentSummaryViz(experiments, eigenval_list)
  summary_viz.save(join(root_out_dir, "Summary.html"), overwrite=True)