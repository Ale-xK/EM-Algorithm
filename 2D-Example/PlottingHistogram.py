import math
import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal

from gendatapt import gen_point
from gendb import complete_database
from gendb import incomplete_database
from Initialize import *

def sep_data(data): #helper function so that plt.hist2d below can work
    lst = []
    lst1 = []
    
    for point in data:
        lst.append(point[0])
        lst1.append(point[1])
    
    return lst,lst1 




    



