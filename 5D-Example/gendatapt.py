import math
import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal

def gen_point(sources):
    '''(list) -> ndarray
    Takes in a list of k sources of unknown distribution. Outputs 1 sample in the form of an array drawn from a source. Source picked from k-nomial distribtion.
    
    >>> station_1 = sp.stats.multivariate_normal([2.5, 2.5, 2.5, 2.5, 2.5] , [[0.5, 0, 0, 0, 0],[0, 0.5, 0, 0, 0], [0, 0, 0.5, 0, 0], [0, 0, 0, 0.5, 0], [0, 0, 0, 0, 0.5]])
    >>> station_2 = sp.stats.multivariate_normal([100, 100, 100, 100, 100], [[50, 0, 0, 0, 0], [0, 50, 0, 0, 0], [0, 0, 50, 0, 0], [0, 0, 0, 50, 0], [0, 0, 0, 0, 50]]) 
    >>> source_list = [station_1, station_2]
    >>>
    >>> gen_point(source_list)
    source number
    1

    [2.94399633 1.86910355 1.7778934  3.31311754 3.68164221]

    array([2.94399633, 1.86910355, 1.7778934 , 3.31311754, 3.68164221,
       1.        ])
    >>>   
    >>> gen_point(source_list)
    source number
    2
    
    [ 97.18625719  97.75299074  99.83867986 106.19662812 110.47246165]
    
    array([ 97.18625719,  97.75299074,  99.83867986, 106.19662812,
           110.47246165,   2.        ])
    '''
    
    k = len(sources)  # number of sources
    
    multinomial_rv = sp.stats.multinomial(1, [1/k]*k) # probabilities vector can be unequal
    
    #multinomial_rv = scipy.stats.multinomial(1000, [0.2, 0.3, 0.5]) # an e.g. for 3 sources
    
    # generate source
    # this will generate an el'tary v with 1 in the place of the chosen source
    source_vector = multinomial_rv.rvs() 
    
    
    # Now we need to extract the number of the chosen source
    for element in source_vector:
        this_source = None
        val = None
        for el in element:
            if el == 1:
                element = element.tolist()
                this_source = element.index(el) 
                val = this_source +1
                 
    # Generate the data point
    chosen_source = sources[this_source] # regardless of distribution, they all take .rvs() method
    this_data_point = chosen_source.rvs() # this line generates the r. vector
    
    complete_data_point = np.append(this_data_point, val) # adds source num to make complete data
    
    return complete_data_point 

                            
# CONCLUSION: this function will simply generate a single data point without creating the db
# THIS IS A HELPER FN
# A separate fn in gendb.py will have to call on this one to create and store 10k pts.