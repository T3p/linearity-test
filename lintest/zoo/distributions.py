"""Distributions to play with"""
import numpy as np

def standard(n, d):
    """n samples from d-dimensional standard normal organized into nxd array"""
    return np.random.normal(size=(n , d))