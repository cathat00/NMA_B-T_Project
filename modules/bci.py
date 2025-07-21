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


  def train(self, proj, target, eigenvectors, targets_by_trial):

    _,_,num_pcs = proj.shape
    inputP = proj[...,:self.n_pcs_to_select]
    
    # -- Estimate BCI weights
    # Regressing target coordinates against neural activity
    # X = Neural activity
    # Y = Predicted target coordinates
    X = np.zeros((inputP.shape[0]*inputP.shape[1], inputP.shape[-1]))
    Y = np.zeros((inputP.shape[0]*inputP.shape[1], self.n_output_units))

    for j in range(inputP.shape[0]): # Fill up X and Y matrices
        X[j*inputP.shape[1]:(j+1)*inputP.shape[1],:] = inputP[j]
        Y[j*inputP.shape[1]:(j+1)*inputP.shape[1],:] = target[targets_by_trial[j]]

    reg = lm.LinearRegression()
    reg.fit(X,Y) # Solve for BCI weights
    W_bci = reg.coef_ # BCI weights
    
    # -- Get transformer matrix: T = D @ P
    P = eigenvectors.real.T
    D = np.zeros((2, num_pcs))
    D[:,:self.n_pcs_to_select] = W_bci
    
    self.transformer = D @ P
