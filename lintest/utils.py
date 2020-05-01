import numpy as np


BALL_RADIUS = 1 / 50 #radius of L2 ball
RTOL = 1e-05 #relative tolerance for closeness checks
ATOL = 1e-08 #absolute tolerance for closeness checks

def count_unique_floats(x, rtol=1e-05, atol=1e-08):
    """
    Counts the unique elements of numpy array x up to the specified tolerance
    See numpy.isclose for the meaning of rtol and atol
    """
    #Check if array has at least two elements
    if x.size < 2: 
        return x.size
    
    #Flat, sorted, unique elements, with no tolerance
    y = np.unique(x)
    
    #Count unique elements with tolerance
    count = 1
    for i in range(1, len(y)):
        if not np.isclose(y[i], y[i-1], rtol, atol):
            count += 1
    
    return count
    

#Sanity checks
if __name__ == '__main__':
    x = np.array([1,2,1+1e-6,2.00001])
    assert count_unique_floats(x) == 2