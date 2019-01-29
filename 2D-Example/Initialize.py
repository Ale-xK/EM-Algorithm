import math
import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt 
import matplotlib as mpl
import seaborn as sns

import sklearn as sk
from sklearn import *

from gendatapt import gen_point
from gendb import complete_database
from gendb import incomplete_database


# Initial parameter estimates - equal weights, unitary covariance, random k means.

# Weights are initialized as equal...
def initialize_weights(sources):
    '''(list) -> list of number
    Takes in a list of k sources and returns a list of k equal weights.
    
    >>> station_1 = sp.stats.multivariate_normal([2.5, 2.5, 2.5, 2.5, 2.5] , [[0.5, 0, 0, 0, 0],[0, 0.5, 0, 0, 0], [0, 0, 0.5, 0, 0], [0, 0, 0, 0.5, 0], [0, 0, 0, 0, 0.5]])
    >>> station_2 = sp.stats.multivariate_normal([100, 100, 100, 100, 100], [[50, 0, 0, 0, 0], [0, 50, 0, 0, 0], [0, 0, 50, 0, 0], [0, 0, 0, 50, 0], [0, 0, 0, 0, 50]]) 
    >>> source_list = [station_1, station_2]
    >>>
    >>> x = initialize_weights(source_list)
    >>> x
    [0.5, 0.5]
    '''
    k = len(sources)
    weights = [1/k] * k

    return weights

# The dimension of the covariance matrix will be lxl, where l is the dimension of the multivariable normal random variable which is the source ...
def initialize_cov(sources):
    '''(list) -> ndarray on number
    Takes in a list of k multivariable sources of dimension l. Returns l-dimensional identity matrix as an ndarray.
    
    >>> station_1 = sp.stats.multivariate_normal([2.5, 2.5, 2.5, 2.5, 2.5] , [[0.5, 0, 0, 0, 0],[0, 0.5, 0, 0, 0], [0, 0, 0.5, 0, 0], [0, 0, 0, 0.5, 0], [0, 0, 0, 0, 0.5]])
    >>>
    >>> station_2 = sp.stats.multivariate_normal([100, 100, 100, 100, 100], [[50, 0, 0, 0, 0], [0, 50, 0, 0, 0], [0, 0, 50, 0, 0], [0, 0, 0, 50, 0], [0, 0, 0, 0, 50]])
    >>>
    >>> source_list = [station_1, station_2]
    >>> x = initialize_cov(source_list)
    >>> x
    array([[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.]])
    '''
    # Checking for errors
    some_mean = sources[0].mean
    some_mean_size = some_mean.size
    
    covariance_matrices = []
    
    for source in sources: 
        if source.mean.size == some_mean_size:
            cov = np.identity(some_mean_size)
            covariance_matrices.append(cov) 
        else:
            print('ERROR: sources are not of equal dimension.')
    return np.asarray(covariance_matrices)

# Initial means are chosen randomly...
def initialize_means(database, sources):
    '''(ndarray, list) -> ndarray
    Takes the incomplete data and the list of sources and returns a list of k means, where k is the number of sources. 
    '''
    k = len(sources)
    means = []
    n = database.shape[0]
    
    for i in range(k):      
        random_db_el = np.random.randint(n)
        means.append(database[random_db_el])

    return np.asarray(means)    