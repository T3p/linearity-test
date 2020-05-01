import numpy as np
import math
from lintest.utils import RTOL, ATOL
from lintest.zoo.distributions import standard
from lintest.testers.additivity import _test_additive, _query_additive

def linearity_tester(f, input_dim, eps, conf=0.9, distr=standard):
    n_samples = math.ceil(2 / eps * math.log(1 / (1 - conf))) + 1
    
    g =  _force_negativity(f, input_dim, eps, conf, distr)
    if type(g) == bool:
        return g
    
    #Additivity
    if not _test_additive(g, input_dim, conf):
        return False
    
    #Epsilon-additivity
    samples = distr(n_samples, input_dim)
    for p in samples:
        answer = _query_additive(p, g, eps, conf)
        if type(answer) == bool:
            return answer
        if not np.allclose(g(p), answer):
            return False
    
    return True

def _force_negativity(f, input_dim, eps, conf, distr):
    n_samples = math.ceil(1 / eps * math.log(1 / (1 - conf)))
    
    x = distr(input_dim, n_samples)
    if not np.allclose(f(-x), -f(x), RTOL, ATOL):
        return False
    
    return lambda x: (f(x) - f(-x)) / 2


#Sanity checks
if __name__ == '__main__':
    def f(x):
        return 2 * x
    
    def g(x):
        return x ** 2
    
    x = np.array([0.3, 1, -2])
    
    ff = _force_negativity(f, 3, 0.1, 0.9, standard)
    assert np.allclose(ff(x), f(x), RTOL, ATOL)
    assert not _force_negativity(g, 3, 0.1, 0.9, standard)
    
    assert linearity_tester(f, 3, 0.1, 0.9) == True
    assert linearity_tester(g, 3, 0.1, 0.9) == False