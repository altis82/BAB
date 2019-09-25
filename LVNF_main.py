import numpy as np
from scipy import optimize
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
#import cvxpy as cvx
import string
import pickle
import plot_LVNF as plot_LVNF
import dump_LVNF as dmp
import plot_settings as ps
#main settings

#running the training file to dump the result
#setting for plot figures

plot_convergence=0
if plot_convergence==1:
    labels=['L-VNFP']
    #def plot_lines_convergence( x_data,datas,numb_of_line, ls,markerstyle,mksize, title, labels, xlb, ylb,stri,xlim,ylim,filename):
    plot_LVNF.plot_lines_convergence([dmp.iteration],[dmp.cost],1,ps.lps,ps.markers,2,'',labels,'Iterations','Total cost',3,[0,3000],[],'convergence_cost')
    

plot_learning_rate=0
if plot_learning_rate==1:
    plot_LVNF.plot_learningrate()
    
plot_totalcost=0
if plot_totalcost==1:
    plot_LVNF.plot_totalcost()

    
plot_instainces=1
if plot_instances==1:
    plot_LVNF.plot_instance()

