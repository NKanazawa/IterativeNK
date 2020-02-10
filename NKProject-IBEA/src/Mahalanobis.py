import numpy as np
import scipy as sp

def calcMahalanobis(C,m,data):
    dist = 0
    diff = data - m
    _,L,U = sp.linalg.lu(C)
    invL = np.linalg.inv(L)
    invU = np.linalg.inv(U)
    invC = np.dot(invU,invL)
    dist = np.dot(np.dot(diff.T,invC),diff)
    return dist