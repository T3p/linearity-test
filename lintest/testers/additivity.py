"""
Distribution-Free Additivity Tester
Fleming and Yoshida, "Distribution-Free Testing of Linear Functions on R^n, (ITCS 2020)
"""
import numpy as np
import math
from lintest.utils import count_unique_floats, BALL_RADIUS, RTOL, ATOL
from lintest.zoo.distributions import standard


def additivity_tester(f, input_dim, eps=0.1, conf=0.9, distr=standard):
    """
    Distribution-free additivity tester
    
    Args:
        f: a vectorized function R^d -> R^m mapping nxd numpy arrays into nxm 
            numpy arrays, where d denotes the domain dimension, m the codomain
            dimension and n the number of points queried at once
        input_dim: dimension of function domain according to the above 
            definition
        eps: closeness parameter
        conf: one minus the probability of a false positive
        distr: input distribution that takes a single size (tuple) parameter
        
    Returns:
        True if the function is eps-close to linear under distr
        Otherwise, False with probability at least conf
    """
    n_samples = math.ceil(2 / eps * math.log(1 / (1 - conf))) + 1
    
    #Additivity
    if not _test_additive(f, input_dim, conf):
        return False
    
    #Epsilon-additivity
    samples = distr((n_samples, input_dim))
    for p in samples:
        answer = _query_additive(p, f, eps, conf)
        if type(answer) == bool:
            return answer
        if not np.allclose(f(p), answer):
            return False
    
    return True

def _test_additive(f, input_dim, conf):
    n_samples = math.ceil(math.log(1 / (1 - conf)) / math.log(100. / 99)) + 1
    
    x = standard((n_samples, input_dim))
    y = standard((n_samples, input_dim))
    z = standard((n_samples, input_dim))
    
    if not np.allclose(f(-x), -f(x), RTOL, ATOL): 
        return False
    if not np.allclose(f(x - y), f(x) - f(y), RTOL, ATOL): 
        return False
    if not np.allclose(f((x - y) / 2), f((x - z) / 2) + f(z - y) / 2, 
                       RTOL, ATOL):
        return False
    
    return True

def _query_additive(p, f, eps, conf):
    n_samples = math.ceil(math.log(2 / eps) / math.log(2))
    
    input_dim = p.shape[-1]
    x = standard((n_samples, input_dim))
    k = _squeezer(p)
    #Check if all the samples give the same result 
    if count_unique_floats(f(p / k - x) + f(x), RTOL, ATOL) > f(p).size:
        return False 
    else: 
        return k * (f(p / k - x[0, :]) + f(x[0, :]))
    
def _squeezer(p):
    if np.linalg.norm(p) <= BALL_RADIUS:
        return 1
    else:
        return math.ceil(np.linalg.norm(p).item() / BALL_RADIUS)

#Sanity checks
if __name__ == '__main__':
    def f(x):
        return 2 * x
    
    def g(x):
        return x ** 2
    
    x = np.array([0.3, 1, -2])
    
    assert _test_additive(f, 5, 0.9) == True
    assert _test_additive(g, 5, 0.9) == False
    
    assert _query_additive(x, f, 0.1, 0.9).size == 3
    assert not _query_additive(x, g, 0.1, 0.9)
    
    assert additivity_tester(f, 3, 0.1, 0.9) == True
    assert additivity_tester(g, 3, 0.1, 0.9) == False