import numpy as np


def count_unique_floats(x, rtol=1e-05, atol=1e-08):
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