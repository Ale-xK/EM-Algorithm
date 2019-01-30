# Five variable EM Algo
 
import math
import numpy as np
import scipy as sp
import pandas as pd
from scipy.stats import multivariate_normal
 
import plotly.plotly as py
import plotly.graph_objs as go 
 
from gendatapt import gen_point
from gendb import complete_database
from gendb import incomplete_database
from Initialize import *
 
 #-------------------------------------DATA-----------------------------------
 
 # The data generated below is used when the EM function is called at the end of this document.
 # The defined sources, their argument values, and the number of data points are all chosen arbitrarily for the sake of demonstration.
 # These values may be changed for experimentation. By 'values' we mean for e.g. arguments for sources, number of sources, number of data points to generate, etc.
 
station_1 = sp.stats.multivariate_normal([2.5, 2.5, 2.5, 2.5, 2.5] , [[0.5, 0, 0, 0, 0],[0, 0.5, 0, 0, 0], [0, 0, 0.5, 0, 0], [0, 0, 0, 0.5, 0], [0, 0, 0, 0, 0.5]])

station_2 = sp.stats.multivariate_normal([4, 4, 4, 4, 4], [[2, 0, 0, 0, 0], [0, 2, 0, 0, 0], [0, 0, 2, 0, 0], [0, 0, 0, 2, 0], [0, 0, 0, 0, 2]])

station_3 = sp.stats.multivariate_normal([1, 1, 3, 3, 3], [[5, 0, 0, 0, 0],[0, 5, 0, 0, 0],[0, 0, 5, 0, 0],[0, 0, 0, 5, 0],[0, 0, 0, 0, 5]])

station_4 = sp.stats.multivariate_normal([ 0, 0, 2, 3, 4], [[1.5, 0, 0, 0, 0],[0, 1.5, 0, 0, 0],[0, 0, 1.5, 0, 0],[0, 0, 0, 1.5, 0],[0, 0, 0, 0, 1.5]]) 
 
source_list = [station_1, station_2, station_3, station_4]

print('\n')
print('GENERATING DATA SET... \n')

x1 = complete_database(source_list, 100)
print('\n','Complete database: \n', x1)

X_train = incomplete_database(x1)
print('\n','Incomplete database: Complete Database minus source info \n', X_train)

#----------------------------------INITIALIZE---------------------------------
print('\n')
print('--------------------------------------------------------------')
print('INITIALIZING PARAMETERS...', '\n') 
 
# It is typical to start with equal weights.
w = initialize_weights(source_list)
print('Initial weights: \n', w)

# It is typical to start with the identity as the covariance.
cov = initialize_cov(source_list)
print('\n','Initial covariance: \n', cov)

# It is typical to choose the initial k-means randomly.
mu = initialize_means(X_train, source_list)
print('\n','Initial means: \n', mu, '\n')

#----------------------------------Algorithm----------------------------------
def EM(data_points, sources, w, mu, cov, iterations, tol):
    """(ndarray, list, list, ndarray, ndarray, int, float) --> None
    
    Takes a (data set of points unassigned to a source), a (list of sources), 
    and parameters to optimize (w, mu, cov), and assigns to each data point a 
    probability of belonging to each source by estimating the parameters: 
    weights, means, and covariances of the respective distributions. 
    The function does this by iterating through the expectation-maximization 
    algorithm 'iterations' number of times or until the log-likelihood of the 
    weighted sum of estimated distributions coverges to within 'tol'.
    At the end of each iteration, the algorithm will update a table with the current estimates of the parameters (weights, means, covariances) for each source.
    This table will be displayed once the algorithm ends.
    Note:- the distributions must be normal,
         - ALL arguments must be defined: (data_points, sources, w, mu, cov, iterations, tol)
    """
    
    k = len(sources)
    num = len(data_points)    
    
    phi = np.zeros((k, num))
    gamma = np.zeros((k, num))
    n = np.zeros((1, k))
    n = n[0] # necessary since the list n creates a list within a list: '[[]]'
    
    # The Main Loop 
    for iteration in range(iterations):    
        
        if iteration == 0:
            cvgce = 10
            
        if (cvgce > tol) and (iteration < iterations):
            print('--------------------------------------------------------------')
            print('Iteration:', iteration+1, '\n')    
        
            iter_params = []  # collect new parameter estimates here before adding to the table      
     
            #------------------------------Expectation----------------------------
            print(' E-step:')
            # Construct phi matrix
            print('Constructing phi matrix...')
            for j in range(k):
                for i in range(num):
                    component = sp.stats.multivariate_normal(mu[j], cov[j]).pdf(X_train[i])
                    phi[j][i] = component
            
            # Initializing log likelihood...           
            # The initialization of the log-likelihood requires the phi matrix,
            # so we held off initializing the log likelihood til now.
            # Calculating value:
            if iteration == 0:
                summ_new = 0
                for i in range(num):
                    inner_sum_new = 0
                    for j in range(k):
                        something_new = w[j] * phi[j][i]
                        inner_sum_new += something_new
                    
                    thing_new = np.log(inner_sum_new)
                    #print('thing_new = ', thing_new)
                    summ_new += thing_new
                
                #print('summ_new = ', summ_new)  
                initial_loglikelihood = float(abs((1/num) * summ_new))            
                print('Initial log likelihood = ', initial_loglikelihood)            
            
            # Construct gamma matrix
            print('Constructing gamma matrix...')
            for i in range(num):
                for j in range(k):
                    denominator = 0
                    for l in range(k):
                        denominator += w[l] * phi[l][i]
                    component = (w[j] * phi[j][i]) / denominator
                    gamma[j][i] = component
            
            # Construct n matrix
            print('Constructing n matrix...')
            for j in range(k):
                component = 0
                for i in range(num):
                    component += gamma[j][i]
                n[j] = component        
    
                
            #------------------------------Maximization-------------------------
            print('\n M-Step:')
            # Construct weights vector
            print('Constructing weights matrix...')
            for j in range(k):
                w[j] = n[j] / num
            iter_params.append(w)
            print('\n', w)
            
            # Construct means matrix
            print('Constructing means matrix...')
            #print('num = ', num)
            for j in range(k):
                sum_j = 0
                for i in range(num):
                    sum_j += gamma[j][i] * X_train[i]
                mu[j] = (1 / n[j]) * sum_j
            iter_params.append(mu)
            print('\n', mu)
            
            # Construct covariance matrix
            print('Constructing covariance matrix...')
            for j in range(k):
                sumj = 0
                for i in range(num):
                    difference = np.asarray([X_train[i] - mu[j]])
                    difference_T = difference.transpose()
                    dotprod = np.dot(difference_T, difference)
                    component = gamma[j][i] * (dotprod)
                    sumj += component
                cov[j] = (1 / n[j]) * sumj
            iter_params.append(cov)
            print('\n', cov)
    
            #----------------------------Convergence----------------------------
            print('\n Convergence Step:')
            
            # recalculate phi matrix
            print('Recalculating phi matrix...')
            for j in range(k):
                for i in range(num):
                    component_new = sp.stats.multivariate_normal(mu[j], cov[j]).pdf(X_train[i])
                    phi[j][i] = component_new   
            
            # determining old_log(-likelihood)
            if iteration == 0:
                old_log = initial_loglikelihood
            else:
                old_log = new_log
                
            # calculating new new_log
            summ_new = 0
            for i in range(num):
                inner_sum_new = 0
                for j in range(k):
                    something_new = w[j] * phi[j][i]
                    inner_sum_new += something_new
                
                thing_new = np.log(inner_sum_new)
                summ_new += thing_new
            new_log = float(abs((1/num) * summ_new))
                
            cvgce = abs(old_log - new_log)
            print('Convergence =', cvgce, '\n')        
        
        elif (cvgce > tol) and (iteration == iterations):
            print('--------------------------------------------------------------')
            print('--------------------------------------------------------------')
            print('--------------------------------------------------------------')
            print('Needs more iterations to achieve desired level of tolerance.')
            print('Iteration Complete.')
            print('--------------------------------------------------------------')
            print('--------------------------------------------------------------')
            print('--------------------------------------------------------------')
            break
            
        else:
            print('--------------------------------------------------------------')
            print('Computation Complete!')
            print('Success after', iteration,'iterations!')
            break
        
        
#---------------------------RUNNING THE ALGORITHM----------------------------
EM(X_train, source_list, w, mu, cov, iterations=10, tol=0.05)
 
 