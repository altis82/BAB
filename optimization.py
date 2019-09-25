import numpy as np
from scipy import optimize
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import cvxpy as cvx
import string
import pickle
import plot_controller as pltmubwater
import dump_controller as dmp
import networkx as nx


#settings create network
V=10
L=range(V,V)
N = 100
b = np.random.random_integers(-2000,2000,size=(N,N))
b_symm = (b + b.T)/2
print b_symm
#create the link between

