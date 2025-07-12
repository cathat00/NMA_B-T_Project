import numpy as np
import sklearn.linear_model as lm 

class BCI():

  def __init__(self, network, target_max, n_output_units=2, scale=0.04, denom=0.2):
    """
    Create feedforward decoder from RNN to (x,y) output units
    for learning (random weights)
    
    """
    # create random weights
    self.decoder = np.random.randn(n_output_units, network.N)
    initial_decoder_fac = scale * (target_max / denom)

    # normalize decoder matrix
    self.decoder *= (initial_decoder_fac / np.linalg.norm(self.decoder))
    self.n_output_units = n_output_units

  def train(self, inputP, target, order):
    # initialize predictor neural activity
    X = np.zeros((inputP.shape[0]*inputP.shape[1], inputP.shape[-1]))

    # initialize predicted target
    Y = np.zeros((inputP.shape[0]*inputP.shape[1], self.n_output_units))

    # fill up
    for j in range(inputP.shape[0]):
        X[j*inputP.shape[1]:(j+1)*inputP.shape[1],:] = inputP[j]
        Y[j*inputP.shape[1]:(j+1)*inputP.shape[1],:] = target[order[j]]

    # regress target against neural activity
    reg = lm.LinearRegression()
    reg.fit(X,Y)

    # make predictions
    y = reg.predict(X)
    mse = np.mean((y-Y)**2)
    return reg.coef_, mse
