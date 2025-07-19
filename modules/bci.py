import numpy as np
import sklearn.linear_model as lm 

class BCI():

  def __init__(self, network, target_max, n_output_units=2, 
               n_pcs_to_select=10, scale=0.04, denom=0.2):
    """
    Create feedforward decoder from RNN to (x,y) output units
    for learning (random weights)
    
    """
    # Initialize decoder matrix
    self.decoder = np.random.randn(n_output_units, network.N)
    initial_decoder_fac = scale * (target_max / denom)
    self.decoder *= (initial_decoder_fac / np.linalg.norm(self.decoder))

    # Initialize transformation matrix
    self.transformer = None

    # Misc. settings
    self.n_output_units = n_output_units
    self.n_pcs_to_select = n_pcs_to_select

  def train(self, proj, eigenvecs, target, order):
    """
    Train the decoder to perform the six-cue
    motor reaching task
    
    """
    # Read projected activity shape
    # (Should be ntrials, tsteps, npcs)
    ntrials, tsteps, npcs = proj.shape

    # Crop projection to n_pcs_to_select eigenvalues
    P = proj[...,:self.n_pcs_to_select]
    
    # Initialize predictor neural activity
    X = np.zeros((P.shape[0]*P.shape[1], P.shape[-1]))
    # Initialize predicted target (screen-space)
    Y = np.zeros((P.shape[0]*P.shape[1], self.n_output_units))

    # Fill up predictor activity and predicted target
    for j in range(P.shape[0]):
        X[j*P.shape[1]:(j+1)*P.shape[1],:] = P[j]
        Y[j*P.shape[1]:(j+1)*P.shape[1],:] = target[order[j]]

    # Fit target to neural activity to estimate decoder
    reg = lm.LinearRegression()
    reg.fit(X,Y)
    self.decoder = np.zeros((self.n_output_units, npcs))
    self.decoder[...,:self.n_pcs_to_select] = reg.coef_

    # Use projection matrix (C) and decoder matrix to compute
    # transformation matrix (T)...
    print(f"Decoder Shape: {self.decoder.shape}")
    print(f"Eigenvecs Shape: {eigenvecs.T.shape}")
    self.transformer = self.decoder @ eigenvecs.T
