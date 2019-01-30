import math
import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from gendatapt import gen_point

# Below are two functions: for generating both complete and incomplete data.

# The reason for creating COMPLETE data is for future testing to see that the eventual EM algo that'll take in the data actually works and sorts the points properly. 

def complete_database(sources, n):
    '''(list, int) -> ndarray
    Takes in a list of k sources of unknown distribution and the number of samples n to produce for the database. Returns n samples in the form of an array drawn from the various source. Source are picked from k-nomial distribtion.
    '''
    complete_db = []
    
    for point in range(n):
        new_data_point = gen_point(sources)
        complete_db.append(new_data_point)
        
    # verifying how the point were distributed:
    k = len(sources)  
    counter = np.zeros(k)
    for point in complete_db:
        source = int(point[-1] - 1)
        counter[source] += 1
    print('Number of data points per source:')
    print(counter)
        
    return np.asarray(complete_db)
        
def incomplete_database(complete_info):
    '''ndarray -> ndarray
    Takes a complete database and removes the source number from each data point. Returns ndarray of incomplete data (each data point has the source number missing).
    '''
    incomplete_db = []
    
    for element in complete_info:
        el = element[:-1]
        incomplete_db.append(el)
        
    return np.asarray(incomplete_db)
    
    