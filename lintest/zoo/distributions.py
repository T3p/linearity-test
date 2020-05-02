"""Distributions to play with and factories to make them"""
import numpy as np

def standard(size):
    """n samples from d-dimensional standard normal organized into nxd array"""
    return np.random.normal(size=size)

def make_uniform(low, high):
    """Curried uniform distribution"""
    def uniform(size):
        return np.random.uniform(low, high, size=size)
    
    return uniform