# Imports for this script
import os
import numpy as np
import modules.configs as configs
import matplotlib.pyplot as plt
from os.path import join
from modules.plots import TaskViz, ManifoldViz, ExperimentSummaryViz

root = "C:/Users/caleb/OneDrive/Desktop/results"
for seed_folder in os.listdir(root):
    seed_results = os.listdir(join(root, seed_folder))
    print(seed_results)
