import pickle
import numpy as np
from os.path import join
from abc import ABC, abstractmethod

# ===============================================
# == UTILITY FUNCS FOR READING / WRITING .PKLs ==
# ===============================================

def save_data(filepath, data):
  with open(filepath, 'wb') as f:
    return pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_data(filepath):
  with open(filepath, 'rb') as f:
      return pickle.load(f)


# =============================================
# == BASE CLASS FOR EXPERIMENT CONFIGURATION ==
# =============================================

class ExperimentConfig(ABC):
  
  def __init__(self, ntrials=80, ntrials_manifold=50, seed=2):
    self.ntrials = ntrials
    self.ntrials_manifold = ntrials_manifold
    self.random_seed = seed
    
    # Set random seed
    if not seed is None:
      np.random.seed(self.random_seed)

  @abstractmethod
  def save(out_dir, **kwargs):
    pass


# ====================================
# == BASIC EXPERIMENT CONFIGURATION ==
# ====================================

class BasicExperimentConfig(ExperimentConfig):
  
  def __init__(self, ntrials=80, ntrials_manifold=50, seed=2):

    super().__init__(ntrials, ntrials_manifold, seed)

    # -- The motor reaching task
    self.task = tasks.BasicReachingTask()
    # -- The recurrent neural network
    self.rnn = RNN(N_in=self.task.ntargets, verbosity=1)
    # -- The brain computer interface
    self.bci = BCI(self.rnn, self.task.target_max)
    # -- The feedback matrix
    self.feedback = np.linalg.pinv(self.bci.decoder)
    
  def save(self, filepath, manifold_data=None):
    data = {
      'params': {
          'random_seed':self.random_seed,
          'ntrials':self.ntrials,
          'ntrials_manifold':self.ntrials_manifold,
      },
      'rnn':self.rnn,
      'task':self.task,
      'bci':self.bci,
      'feedback':self.feedback,
      'manifold_data':manifold_data,
    }
    save_data(filepath, data)
