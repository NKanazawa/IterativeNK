import numpy as np
from scipy.linalg import lu

def calcMahalanobis(C,m,data):
    dist = 0
    diff = data - m
    _,L,U = lu(C)
    invL = np.linalg.inv(L)
    invU = np.linalg.inv(U)
    invC = np.dot(invU,invL)
    dist = np.dot(np.dot(diff.T,invC),diff)
    return dist