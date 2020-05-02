"""Functions to play with and factories to make them"""
import numpy as np

class Oracle:
    """Decorator to keep track of function calls"""
    def __init__(self, f):
        self.f = f
        self._calls = 0
    
    def __call__(self, x):
        self._calls += 1
        return self.f(x)
    
    def reset_calls(self):
        self._calls = 0
    
    def get_calls(self):
        return self._calls

def make_linear(w=1):
    """Factory for linear function <w,x>"""
    @Oracle
    def linear(x):
        return np.dot(x, w)
    
    return linear

def make_noisy_linear(w=1, std=1):
    """Factory for linear function <w,x> perturbed by gaussian noise N(0,std^2)"""
    @Oracle
    def noisy_linear(x):
        return np.dot(x, w) + np.random.normal(scale=std)
    
    return noisy_linear

def make_corrupted_linear(w=1, p=0.1):
    """Factory for linear function <w,x> with p probability of returning zero instead"""
    @Oracle
    def corrupted_linear(x):
        return np.dot(x, w) * (1 - np.random.binomial(1, p))
    
    return corrupted_linear

def make_stretched_tanh(w=1, stretch=1):
    """Factory for tanh(<w,x>/stretch) function"""
    @Oracle
    def stretched_tanh(x):
        return np.tanh(np.dot(x, w) / stretch)
    return stretched_tanh


#Sanity checks
if __name__ == '__main__':
    from lintest.testers.linearity import linearity_tester
    w = np.array([1., -1., 0.5])
    
    g = make_stretched_tanh(w, 1e6)
    assert linearity_tester(g, 3) == True
    
    h = make_linear()
    l = make_linear()
    assert linearity_tester(h, 2) == True
    assert linearity_tester(l, 100) == True
    assert h.get_calls() == l.get_calls()
    h.reset_calls()
    assert h.get_calls() == 0