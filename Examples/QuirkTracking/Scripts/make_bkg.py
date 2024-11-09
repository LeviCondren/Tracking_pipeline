from scipy import fsolve
import numpy as np
import pandas as pd

fourierDimTrain = 2
def sample_spherical(npoints = 2 * fourierDimTrain, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

for i in range(-2,2):
    fc = np.cos(2*np.pi*n*t/Lambda)-1
