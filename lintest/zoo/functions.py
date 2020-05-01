"""Functions to play with and factories to make them"""
import numpy as np

def make_corrupted_linear(w=1, std=1):
    def corrupted_linear(x):
        return np.dot(x, w) + np.random.normal(scale=std)
    
    return corrupted_linear

def make_stretched_tanh(w=1, stretch=1):
    def stretched_tanh(x):
        return np.tanh(np.dot(x, w) / stretch)
    return stretched_tanh


#Sanity checks
if __name__ == '__main__':
    from lintest.testers.linearity import linearity_tester
    w = np.array([1., -1., 0.5])
    f = make_corrupted_linear(w, 1e-9)
    assert linearity_tester(f, 3) == True
    
    g = make_stretched_tanh(w, 1e6)
    assert linearity_tester(g, 3) == True