import numpy as np
from abc import ABC, abstractmethod

#trial

# ===========================================
# == ABSTRACT BASE CLASS FOR TASK CREATION ==
# ===========================================
# This allows us to define common functionality
# for all motor reaching tasks. All task classes 
# should inherit from this class and implement the 
# abstract methods _create_stimuli and _create_targets.
# For more on abstract base classes in Python...
# https://www.datacamp.com/tutorial/python-abstract-classes

class MotorReachingTask(ABC):

  def __init__(self, tsteps:int=200, ntargets:int=6, target_max=0.2,
               stim_length:int=20, stim_amplitude:float=1, 
               stim_is_2d:bool=False):
    # -- Time component
    self.tsteps = tsteps # Timesteps

    # -- Stimulus
    self.stim_length = stim_length # Length of the pulse stimulus (ms)
    self.stim_amplitude = stim_amplitude # Amplitude of the pulse stimulus
    self.stim_is_2d = stim_is_2d

    # -- Targets 
    self.ntargets = ntargets # Number of reaching targets
    self.target_max = target_max # Max X and Y coordinate of target

    # -- Create task
    # Targets shape is (ntargets, stim_length, 2D coordinates)
    self._create_targets()
    # Stimulus shape is (ntargets, stim_length, ntargets)
    self._create_stimuli() 

  @abstractmethod
  def _create_stimuli(self):
    pass

  @abstractmethod
  def _create_targets(self):
    pass


# =========================
# == BASIC REACHING TASK ==
# =========================
# The basic reaching task, used by Feulner.

class BasicReachingTask(MotorReachingTask):

  def _create_stimuli(self):
    self.stimuli = np.zeros((self.ntargets, self.tsteps, self.ntargets))
    if self.stim_is_2d:
        phis = np.linspace(0,2*np.pi,self.targets,endpoint=False)
        for j in range(self.stimuli.shape[0]):
            self.stimuli[j,:self.stim_length,0] = self.stim_amplitude*np.cos(phis[j])
            self.stimuli[j,:self.stim_length,1] = self.stim_amplitude*np.sin(phis[j])
            self.stimuli[j,:self.stim_length,2:] = 0
    else:
        for j in range(self.ntargets):
            self.stimuli[j,:self.stim_length,j] = self.stim_amplitude

  def _create_targets(self):
    # create target trajectories
    phis = np.linspace(0, 2*np.pi, self.ntargets, endpoint=False)
    rs = np.zeros(self.tsteps)

    # define each target's x and y coordinate
    rs[self.stim_length:] = np.ones(self.tsteps-self.stim_length)*self.target_max
    traj = np.zeros((self.ntargets,self.tsteps,2))
    for j in range(self.ntargets):
        # create x-coordinate on screen
        traj[j,:,0] = rs*np.cos(phis[j])
        # create y-coordinate on screen
        traj[j,:,1] = rs*np.sin(phis[j])
    self.targets = traj


class GeometricReachingTask(MotorReachingTask):

  def _create_stimuli(self):
    self.stimuli = np.zeros((self.ntargets, self.tsteps, self.ntargets))
    if self.stim_is_2d:
        phis = np.linspace(0,2*np.pi,self.targets,endpoint=False)
        for j in range(self.stimuli.shape[0]):
            self.stimuli[j,:self.stim_length,0] = self.stim_amplitude*np.cos(phis[j])
            self.stimuli[j,:self.stim_length,1] = self.stim_amplitude*np.sin(phis[j])
            self.stimuli[j,:self.stim_length,2:] = 0
    else:
        for j in range(self.ntargets):
            self.stimuli[j,:self.stim_length,j] = self.stim_amplitude

  def _create_targets(self):
    # create target trajectories
    phis = np.linspace(0, 2*np.pi, self.ntargets, endpoint=False)
    rs = np.zeros(self.tsteps)

    # define each target's x and y coordinate
    rs[self.stim_length:] = np.ones(self.tsteps-self.stim_length)*self.target_max
    traj = np.zeros((self.ntargets,self.tsteps,2))
    for j in range(self.ntargets):
        # create x-coordinate on screen
        traj[j,:,0] = rs*np.cos(phis[j])
        # create y-coordinate on screen
        traj[j,:,1] = rs*np.sin(phis[j])
    self.targets = traj
