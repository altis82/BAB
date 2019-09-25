# -*- coding: utf-8 -*-
"""
Filename: plot_VNF.py
Authors: Chuan Pham
"""
from __future__ import division

import numpy as np

import matplotlib.pyplot as plt
plt.rcParams['legend.fancybox'] = True
import plot_settings as ps
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
import dump_LVNF as dmp
importlib.reload(ps)



num_days = 1
scale = 100
markers = ['o','x', 's','v','d','^','<','>','+','*','x']
linestyles = ['-','-','-','-']
linewidth = 1.5
colors= ['lightblue','navy','teal','darkorange', 'r','purple', 'g', 'k', 'dodgerblue', 'm','darksalmon','b','y','darkgrey','c']
new_colors=['#446CB3','#FABE58','#E74C3C','#C8F7C5']#red,#
start_iter = 0
runTime = 100
maxtimeslot=100
stride=2



#plot learning rate

def plot_learningrate():
    ps.set_mode("tiny")
    fig=plt.gcf()
    x_data=[dmp.time1,dmp.time2,dmp.time3]
    y_data=[dmp.error1,dmp.error2,dmp.error3]
    numb_of_lines=3
    labels=['Learning rate 0.01','Learning rate 0.007','Learning rate 0.004']
    for line in range(numb_of_lines):
        print(line)
        #plt.scatter(x_data[line],y_data[line], color=colors[line],ls=linestyles[line], label=labels[line])
        plt.plot(x_data[line],y_data[line], color=colors[line],ls=linestyles[line], label=labels[line],markevery=1)
    plt.legend(loc=1)
   
    #plt.yscale('log')
    
    plt.xlabel('Seconds')
    plt.ylabel('Mean square error')
    plt.show()
    fig.tight_layout()
    name='plot_learningrate.pdf'
    fig.savefig(name)
        
def plot_lines_convergence( x_data,datas,numb_of_line, ls,markerstyle,mksize, title, labels, xlb, ylb,stri,xlim,ylim,filename):
    ps.set_mode("tiny")
    fig=plt.gcf()
    #stride=2
    data=datas[0]
    # for i in range(len(data)):
    #     if i>200:
    #         data[i]=data[i]/(i*0.0002+1)
    for line in range(numb_of_line):
        plt.plot(x_data[line],data, color='teal',ls='-', label=labels[line],markevery=stri)
    #plot optimal
    optimal=np.ones(len(dmp.iteration))*17.4
    plt.plot(dmp.iteration,optimal,ls=':',label='Optimal',color='red')
    
    plt.legend(loc=1)
    if xlim != []:
        plt.xlim(xlim)
    if ylim!=[]:
        plt.ylim(ylim)
    if title!='':
        plt.title(title)
    #plt.yscale('log')
    fig=plt.gcf()
    plt.xlabel(xlb)
    plt.ylabel(ylb)
    
    plt.axes([.5, .45, .35, .25])
    zoomx=np.zeros(20)
    zoomy=np.zeros(20)
    for i in range(20):
        j=220+i
        zoomx[i]=dmp.iteration[j]
        zoomy[i]=dmp.cost[j]
    
    plt.plot(zoomx,zoomy,color='teal')
    plt.yticks([17.843,17.85])

    
    plt.show()
    fig.tight_layout()
    name=filename+'.pdf'
    fig.savefig(name)
    
def plot_totalcost():
    labels=['Branch-and-bound','RMINLP','L-VNFP']
    ps.set_mode("tiny")
    fig = plt.gcf()
    number_lines=3
    x_data=dmp.optimal_nodes
    #print(len(x_data))
    y_data=[dmp.optimal_cost,dmp.rminlp_cost,dmp.lvnf_cost]
    for i in range(number_lines):        
        plt.plot(x_data,y_data[i], color=new_colors[i],ls=linestyles[i], marker=markers[i],label=labels[i],markevery=5)
    plt.ylabel('Total cost')
    plt.xlabel('Number of nodes')
    plt.legend(loc=2)
    plt.show()
    
    filename="plot_totalcost"
    fig.tight_layout()
    name = filename + '.pdf'
    fig.savefig(name)
    
    plt.show()
    
    filename="plot_instance"
    fig.tight_layout()
    name = filename + '.pdf'
    fig.savefig(name)
def plot_instance:
    ps.set_mode("tiny")
    fig = plt.gcf()
    N = len(dmp.numberof_CT_dcsm)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.25  # the width of the bars: can also be len(x) sequence
    p1 = plt.bar(ind, np.array(dmp.Energy_cost_dcsm), width, color=new_colors[0], align='center')
    p2 = plt.bar(ind + width, np.array(dmp.Energy_cost_shortest), width, color=new_colors[1], align='center')
    p3 = plt.bar(ind + 2 * width, np.array(dmp.Energy_cost_smt), width, color=new_colors[2], align='center')
    plt.ylabel('Energy cost')
    plt.xticks(ind + width - 0.15, dmp.numberof_SC)
    # plt.yticks(np.arange(0, 81, 10))
    plt.legend((p1[0], p2[0], p3[0]), ('DCSM', 'Shortest', 'SMT'), loc=2, ncol=1)

def plot_cost_case1():
    ps.set_mode("tiny")
    labels = ['DCSM', 'SMT', 'K-mean']
    markerstyle = ['>', 'o', '+', '+', 's', 'd', '^', '<', '>', '+', '*', 'x']
    fig = plt.gcf()
    N = len(dmp.numberof_CT_dcsm)
    ind = np.arange(20,60, step=5)  # the x locations for the groups
    width = 0.25  # the width of the bars: can also be len(x) sequence

    data = [dmp.numberof_CT_dcsm, dmp.numberof_CT_SMT, dmp.numberof_CT_Shortestpath]
    numb_of_line=3
    for line in range(numb_of_line):
        plt.plot(data[line], color=new_colors[line], ls=linestyles[line], marker=markers[line], markersize=4, label=labels[line],
                 markevery=1)

    plt.ylabel('\# controllers')
    plt.xlabel('\# service chains')

    plt.xticks(np.arange(0,40, step=5),np.arange(20,60, step=5))
    plt.ylim([2,8.5])
    plt.legend( ('DCSM','SMT','K-mean'),loc=1,ncol=3)

    plt.show()
    filename="plot_num_CT_case1"
    fig.tight_layout()
    name = filename + '.pdf'
    fig.savefig(name)

    # ps.set_mode("tiny")
    # fig = plt.gcf()
    # N = len(dmp.numberof_CT_dcsm)
    # ind = np.arange(N)  # the x locations for the groups
    # width = 0.25  # the width of the bars: can also be len(x) sequence
    # p1 = plt.bar(ind, np.array(dmp.Energy_cost_dcsm), width, color=new_colors[0], align='center')
    # p2 = plt.bar(ind + width, np.array(dmp.Energy_cost_shortest), width, color=new_colors[1], align='center')
    # p3 = plt.bar(ind + 2 * width, np.array(dmp.Energy_cost_smt), width, color=new_colors[2], align='center')
    #
    # plt.ylabel('Energy cost')
    # plt.xticks(ind + width - 0.15, dmp.numberof_SC)
    # # plt.yticks(np.arange(0, 81, 10))
    # plt.legend((p1[0], p2[0], p3[0]), ('DCSM', 'Shortest', 'SMT'), loc=2, ncol=1)

    ps.set_mode("tiny")
    labels = ['DCSM', 'SMT', 'K-mean']
    fig=plt.gcf()
    numb_of_line=3
    data=np.array([dmp.Energy_cost_dcsm,dmp.Energy_cost_smt,dmp.Energy_cost_shortest])*.14
    for line in range(numb_of_line):
        plt.plot(data[line], color=new_colors[line],ls='-',marker = markers[line], markersize=0, label=labels[line],markevery=1)
    plt.xlabel('\# service chains')
    plt.xticks(ind + width - 0.15, dmp.numberof_SC)
    plt.ylabel('Energy cost')
    plt.xticks(np.arange(0, len(dmp.Energy_cost_dcsm), step=5), np.arange(20, 60, step=5))

    plt.xlabel('\# service chains')
    plt.legend( ('DCSM', 'SMT', 'K-mean'), loc=4, ncol=1)

    plt.xlim([0, 40])

    plt.show()


    filename = "plot_cost_case1"
    fig.tight_layout()
    name = filename + '.pdf'
    fig.savefig(name)

    ps.set_mode("tiny")
    fig, ax = plt.subplots()
    width = 0.3
    numberofsite=3
    patterns = ["|||", "\\", "///", "...", "---", "xxx", "+++", "o", "O", "*"]
    name = ['DCSM', 'SMT', 'K-mean']
    index = np.arange(7)
    # for i in range(numberofsite):
    i = 0
    data=[[.67,.7,.7,.61,.6,.62,.72],[.85,.6,.85,.6,.78,.82,.83],[.89,.89,.5,.6,.5,.87,.87]]
    for i in range(3):
        ax.bar(index + width * i, data[i], width=width, alpha=0.4, color=new_colors[i], edgecolor='black')

    # ax.set_ylim([10., 350])
    ax.legend(name,loc=1,ncol=3)

    ax.set_ylim([.4,1])
    ax.set_ylabel("Utilization")
    ax.set_xlabel("Controller")
    plt.xticks(index + width, ([1,2,3,4,5,6,7]))
    fig.tight_layout()
    plt.savefig('plot_utilization.pdf')
    plt.show()


def plot_cost_case2():
    ps.set_mode("tiny")
    labels = ['DCSM', 'SMT', 'K-mean']
    markerstyle = ['>', 'o', '+', '+', 's', 'd', '^', '<', '>', '+', '*', 'x']
    fig = plt.gcf()
    N = len(dmp.numberof_CT_dcsm)
    ind = np.arange(20,60, step=5)  # the x locations for the groups
    width = 0.25  # the width of the bars: can also be len(x) sequence

    data = [dmp.total_cost_dcsm_case2, dmp.total_cost_smt_case2, dmp.total_cost_shortest_case2]
    numb_of_line=3
    for line in range(numb_of_line):
        plt.plot(data[line], color=new_colors[line], ls=linestyles[line], marker=markers[line], markersize=4, label=labels[line],
                 markevery=1)

    plt.ylabel('Total cost')
    plt.xlabel('\# service chains')

    plt.xticks(np.arange(0,40, step=5),np.arange(20,60, step=5))
    #plt.ylim([2,8.5])
    plt.legend( ('DCSM','SMT','K-mean'),loc=2,ncol=3)

    plt.show()
    filename="plot_cost_case2"
    fig.tight_layout()
    name = filename + '.pdf'
    fig.savefig(name)
def plot_correlation():
    ps.set_mode("tiny")
    fig, ax = plt.subplots()
    cax = ax.imshow(np.array(dmp.correlation_dcsm), interpolation='nearest', cmap=cm.afmhot)
    cbar = fig.colorbar(cax, ticks=[0, 0.2,0.4], orientation='vertical')
    cbar.ax.set_xticklabels(['Low', 'Medium', 'High'])  # horizontal colorbar
    filename = "plot_correlation_dcsm"
    plt.xlabel("Controller")
    plt.ylabel("Controller")
    plt.show()
    fig.tight_layout()
    name = filename + '.pdf'
    fig.savefig(name)
    ############################
    fig, ax = plt.subplots()
    cax = ax.imshow(np.array(dmp.correlation_shortest), interpolation='nearest', cmap=cm.afmhot)
    cbar = fig.colorbar(cax, ticks=[0, 0.2, 0.4], orientation='vertical')
    cbar.ax.set_xticklabels(['Low', 'Medium', 'High'])  # horizontal colorbar
    filename = "plot_correlation_shortest"
    plt.xlabel("Controller")
    plt.ylabel("Controller")
    plt.show()
    fig.tight_layout()
    name = filename + '.pdf'
    fig.savefig(name)

    ############################
    fig, ax = plt.subplots()
    cax = ax.imshow(np.array(dmp.correlation_smt), interpolation='nearest', cmap=cm.afmhot)
    cbar = fig.colorbar(cax, ticks=[0, 0.2, 0.4], orientation='vertical')
    cbar.ax.set_xticklabels(['Low', 'Medium', 'High'])  # horizontal colorbar
    filename = "plot_correlation_smt"
    plt.xlabel("Controller")
    plt.ylabel("Controller")
    plt.show()
    fig.tight_layout()
    name = filename + '.pdf'
    fig.savefig(name)
def plot_everage_delay(data, numberofsite):
    ps.set_mode("tiny")
    fig, ax = plt.subplots()
    width = 0.3
    patterns = ["|||", "\\", "///", "...", "---", "xxx", "+++", "o", "O", "*"]
    name = ['DCSM', 'SMT', 'K-mean']
    index = np.arange(numberofsite)
    #for i in range(numberofsite):
    i=0
    ax.bar(index + width * i, data, width=width, alpha=0.4, color='red', edgecolor='black')

    # ax.set_ylim([10., 350])
    ax.set_ylabel("Latency (ms)")
    plt.xticks(index +0.2, ('DCSM', 'SMT', 'K-mean'))
    fig.tight_layout()
    plt.savefig('plot_everage_delay.pdf')
    plt.show()
def plot_cdf():
    labels = ['R-DCSM', 'SMT','K-mean']
    ps.set_mode("tiny")
    fig = plt.gcf()
    N = 3
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence
    x_data=[dmp.RDCSM_CDF_iteration,dmp.SMT_CDF_iteration,dmp.DCA_iteration]
    data=[dmp.RDCSM_CDF,dmp.SMT_CDF,dmp.DCA_CDF]
    numb_of_line=3
    for line in range(numb_of_line):
        plt.plot(x_data[line],data[line], color=ps.colors[line],ls='-', label=labels[line],markevery=2)

    plt.legend(loc=4)
    # if xlim != []:
    #     plt.xlim(xlim)
    # if ylim != []:
    #     plt.ylim(ylim)
    # if title != '':
    #     plt.title(title)
    filename="plot_CDF"
    plt.xlabel("Iterations")
    plt.ylabel("CDF")
    plt.show()
    fig.tight_layout()
    name = filename + '.pdf'
    fig.savefig(name)

    labels = ['DCSM','R-DCSM','D-DCSM', 'SMT', 'K-mean']
    ps.set_mode("tiny")
    fig = plt.gcf()
    N = 3
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence
    x_data = [20,60,110]
    data = [dmp.dcsm_time, dmp.rdcsm_time,dmp.ddcsm_time, dmp.smt_time,dmp.short_time]
    numb_of_line = 5
    markerstyle = ['>', '*', 'o', '+', 's', 'd', '^', '<', '>', '+', '*', 'x']
    for line in range(numb_of_line):
        plt.plot(x_data, data[line], color=ps.colors[line], ls='-', label=labels[line], markevery=1)

    plt.legend(loc=2)
    # if xlim != []:
    #     plt.xlim(xlim)
    # if ylim != []:
    #     plt.ylim(ylim)
    # if title != '':
    #     plt.title(title)
    filename = "plot_execution_time"
    plt.xlabel("\# service chains")
    plt.ylabel("Time (seconds)")
    plt.show()
    fig.tight_layout()
    name = filename + '.pdf'
    fig.savefig(name)


def plot_total_cost( x_data,datas,numb_of_line, ls,markerstyle,mksize, title, labels, xlb, ylb,stri,xlim,ylim,filename):
    ps.set_mode("small")
    fig=plt.gcf()
    colors = ['orange','lightblue','darkgreen','gold', 'r', 'navy', 'teal', 'darkorange', 'purple', 'g', 'k', 'dodgerblue', 'm', 'darksalmon', 'b',
              'y', 'darkgrey', 'c']
    linestyles = ['-', '-','-','-.','-']  # ,'-.']
    #stride=2
    for line in range(numb_of_line):
        plt.plot(x_data[0],datas[line], color=colors[line],ls=linestyles[line],marker = markerstyle[line], markersize=mksize, label=labels[line],markevery=stri)
    plt.legend(loc=1,ncol=3,fontsize ='medium')
    if xlim!=[]:
        plt.xlim(xlim)
    #plt.ylim([35,160])
    if title!='':
        plt.title(title)
    plt.xlabel(xlb)
    plt.ylabel(ylb)
    fig = plt.gcf()
    #    x = [25,19.5]
    #    y = [200,410]
    x = [8, 3.8]
    y = np.array([38, 55])
    plt.plot(x, y, 'b-', linewidth=1)
    #    x = [5,5]
    #    y = [200,410]
    x = [9, 6.5]
    y = np.array([42, 55])
    plt.plot(x, y, 'b-', linewidth=1)

    plt.axes([.2, .55, .3, .15], axisbg='w')
    #optimal
    plt.plot(np.array([286.5,	303.5])/8, color='r')
    #rounding
    plt.plot(np.array([318.5,	339.5,])/8,ls='-.', color='gold')
    #dcms
    plt.plot(np.array([298.5,	315.5])/8, color='orange')
    plt.xticks([0,1],[8,9])
    plt.yticks([30,35, 40])
    plt.show()
    fig.tight_layout()
    name=filename+'2.pdf'
    fig.savefig(name)

def plot_lines_axis( x_data,datas,numb_of_line, ls,markerstyle,mksize, title, labels, xlb, ylb,stri,xlim,ylim,filename):
    ps.set_mode("small")
    fig=plt.gcf()
    #stride=2
    for line in range(numb_of_line):
        plt.plot(x_data[line],datas[line], color=ps.colors[line],ls=ls[line],marker = markerstyle[line], markersize=mksize, label=labels[line],markevery=stri)
    plt.legend(loc=1)
    if xlim!=[]:
        plt.xlim(xlim)
    if ylim!=[]:
        plt.ylim(ylim)
    if title!='':
        plt.title(title)
    plt.yscale('symlog')
    plt.xlabel(xlb)
    plt.ylabel(ylb)
    plt.show()
    fig.tight_layout()
    name=filename+'.pdf'
    fig.savefig(name)

def plot_bar_energy(data,numofsite):
    ps.set_mode("small")
    fig = plt.gcf()

    ax = fig.add_subplot(111, projection='3d')
    i=0
    width=0.15
    index = np.arange(numofsite)
    for c, z in zip(['r', 'g', 'b', 'y', 'c'], [1,2,3,4,5]):
        xs = np.arange(9)
        ys = data[i]
        i=i+1
        # You can provide either a single color or an array. To demonstrate this,
        # the first bar of each set will be colored cyan.
        cs = [c] * len(xs)

        ax.bar(xs, ys, zs=z, zdir='y', color=cs, alpha=0.8)
        if i==4:
            plt.yticks(index + width + 1, ('1', ' 2', '3', '4', '5'))

    ax.set_xlabel('Time slot')
    ax.set_ylabel('MUB')
    ax.set_zlabel('Energy (KWh)')
    plt.show()
    fig.tight_layout()
    fig.savefig('plot_bar_energy.pdf')

# def plot_bar_water(data,numofsite):
#     ps.set_mode("small")
#     fig = plt.gcf()
#
#     ax = fig.add_subplot(111, projection='3d')
#     i=0
#     width=0.15
#     index = np.arange(numofsite)
#     for c, z in zip(['r', 'g', 'b', 'y', 'c'], [1,2,3,4,5]):
#         xs = np.arange(9)
#         ys = data[i]
#         i=i+1
#         # You can provide either a single color or an array. To demonstrate this,
#         # the first bar of each set will be colored cyan.
#         cs = [c] * len(xs)
#
#         ax.bar(xs, ys, zs=z, zdir='y', color=cs, alpha=0.8)
#         if i==4:
#             plt.yticks(index + width + 1, ('1', ' 2', '3', '4', '5'))
#
#     ax.set_xlabel('Time slot')
#     ax.set_ylabel('MUB')
#     ax.set_zlabel('Water (L)')
#     plt.show()
#     fig.tight_layout()
#     fig.savefig('plot_bar_water.pdf')
def plot_energy(data,numofsite):
    #Create  figure.
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Generate example data.
    R, Y = np.meshgrid(np.arange(1, numofsite+1, 1), np.arange(1, 9+1, 1))
    z = 0.1 * np.abs(np.sin(R / 40) * np.sin(Y / 6))
    for i in range(9):
        for j in range(numofsite):
            z[i][j]=np.transpose(data)[i][j]
    #print z


    #print z
    # Plot the data.
    surf = ax.plot_surface(R, Y, z, cmap=cm.jet, linewidth=0)
    fig.colorbar(surf)

    # Set viewpoint.
    ax.azim = -160
    ax.elev = 30

    # Label axes.
    ax.set_xlabel('Along track (m)')
    ax.set_ylabel('Range (m)')
    ax.set_zlabel('Height (m)')
    fig.show()
    # Save image.
    fig.savefig('energy.pdf')
def plot_energy_bar(data, numberofsite):
    ps.set_mode("small")
    fig, ax = plt.subplots()
    width = 0.15
    patterns = ["|||", "\\", "///", "...", "---", "xxx", "+++", "o", "O", "*"]
    name = ['MUB 1', 'MUB 2', 'MUB 3', 'MUB 4', 'MUB 5']

    # ax.annotate('The comfort energy of \n office tenant 1 at MUB 3.', xy=(2.2, 200), xytext=(3, 250.5),
    #             arrowprops=dict(facecolor='black', width=0.1, frac=0.2, headwidth=3),
    #             )

    index = np.arange(numberofsite)
    for i in range(numberofsite):
        ax.bar(index + width * i, data[i], width=width, alpha=0.4, color=ps.colors[i], edgecolor='black')

    #ax.set_ylim([10., 350])
    ax.set_ylabel("Energy (kW)")
    plt.xticks(index + width + 0.05, ('MUB 1', 'MUB 2', 'MUB 3', 'MUB 4', 'MUB 5'))
    fig.tight_layout()
    plt.savefig('plot_energy_lim.pdf')
    plt.show()
def plot_bar_battery(datas, num_of_bars,numb_of_cols, filename):
    ps.set_mode("tiny")
    fig = plt.gcf()
    ax = fig.add_subplot(111)
    ind = np.arange(numb_of_cols)  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence
    bottomdt=np.zeros(numb_of_cols)
    p1 = ax.bar([0.5,1.85], datas[0], width, color='dodgerblue',align='center')
    #for i  in range(2):
    bottomdt = bottomdt + datas[0]
        #print bottomdt
        # for j in range(numb_of_cols):
        #     bottomdt[j]=bottomdt[i]+datas[i][j]
    p1 = ax.bar([0.5,1.85], datas[1], 0.35, color='r',bottom = bottomdt,align='center')
        #plt.show()
    plt.xticks([0.55,1.85],('MUB4','MUB5'))
    plt.ylabel('Power (KW)')
    plt.title('')
    #plt.xlim(0,2)
    #plt.xticks(ind, ind)


    plt.legend(('Battery', 'Fue'),loc=4,ncol=2)

    plt.show()
    fig.tight_layout()
    name = filename + '.pdf'
    fig.savefig(name)
def plot_compare_energy(datas, num_of_bars, numb_of_cols, filename):
    ps.set_mode("tiny")
    fig = plt.gcf()
    ax = fig.add_subplot(111)
    ind = np.arange(numb_of_cols)  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence
    bottomdt = np.zeros(numb_of_cols)
    #(31, 119, 180), (174, 199, 232),
    colors=[ (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    for i in range(len(colors)):
        r, g, b = colors[i]
        colors[i] = (r / 255., g / 255., b / 255.)
    p1 = ax.bar(ind, datas[0], width, color=colors[0])
    for i in range(numb_of_cols-1):
        bottomdt = bottomdt + datas[i]
        # print bottomdt
        # for j in range(numb_of_cols):
        #     bottomdt[j]=bottomdt[i]+datas[i][j]
        p1 = ax.bar(ind, datas[i + 1], width, color=colors[i + 1])
        # plt.show()
    plt.ylim([0,75])
    plt.ylabel('Energy (MWh)')
    plt.title('')
    plt.xticks(ind+width/2, ["JEWAS-ON","Uncordinated","Over"])
    # plt.yticks(np.arange(0, 81, 10))

    plt.legend(('DC', 'HVAC', 'Battery', 'Fuel'), loc=2, ncol=3)

    plt.show()
    fig.tight_layout()
    name = filename + '.pdf'
    fig.savefig(name)


def plot_compare_water(datas, num_of_bars, numb_of_cols, filename):
    ps.set_mode("tiny")
    fig = plt.gcf()
    ax = fig.add_subplot(111)
    ind = np.arange(numb_of_cols)  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence
    bottomdt = np.zeros(numb_of_cols)
    #(31, 119, 180), (174, 199, 232),
    colors=[
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    for i in range(len(colors)):
        r, g, b = colors[i]
        colors[i] = (r / 255., g / 255., b / 255.)

    p1 = ax.bar(ind, datas[0], width, color=colors[0])
    p1 = ax.bar(ind+width, datas[1], width, color='r')


    plt.ylabel('Water (L)')
    plt.xticks(ind+width, ["MUB1","MUB2","MUB3","MUB4","MUB5"])
    # plt.yticks(np.arange(0, 81, 10))

    plt.legend(('JEWAS-ON', 'Over'), loc=2, ncol=2)

    plt.show()
    fig.tight_layout()
    name = filename + '.pdf'
    fig.savefig(name)

def plot_bar_battery_charging(datas, num_of_bars, numb_of_cols, filename):
    ps.set_mode("tiny")
    fig = plt.gcf()
    ax = fig.add_subplot(111)
    ind=np.arange(len(datas[0]))
    width = 0.35  # the width of the bars: can also be len(x) sequence
    p1 = ax.bar(ind,datas[0], width, color='b', align='edge')
    p1 = ax.bar(ind, datas[1], width, color='r', align='edge')

    plt.xlabel('Time slot')
    plt.ylabel('Power (KW)')
    plt.ylim([-700,700])
    plt.legend(['Discharging','Charging'], loc=2, ncol=2)

    plt.show()
    fig.tight_layout()
    name = filename + '.pdf'
    fig.savefig(name)
def plot_bar(datas, num_of_bars,numb_of_cols, filename):
    ps.set_mode("small")
    fig = plt.gcf()
    ax = fig.add_subplot(111)
    ind = np.arange(numb_of_cols)  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence
    bottomdt=np.zeros(numb_of_cols)
    p1 = ax.bar(ind, datas[0], width, color=ps.colors[0])
    for i  in range(4):
        bottomdt = bottomdt + datas[i]
        #print bottomdt
        # for j in range(numb_of_cols):
        #     bottomdt[j]=bottomdt[i]+datas[i][j]
        p1 = ax.bar(ind, datas[i+1], width, color=ps.colors[i+1],bottom = bottomdt)
        #plt.show()
    plt.xlabel('Time slots')
    plt.ylabel('Workloads')
    plt.title('')
    #plt.xticks(ind, ind)
    #plt.yticks(np.arange(0, 81, 10))

    plt.legend(('DC 1', 'DC 2','DC 3','DC 4','DC 5'),loc=2,ncol=2)

    plt.show()
    fig.tight_layout()
    name = filename + '.pdf'
    fig.savefig(name)
def plot_lines(datas, numb_of_line, ls,markerstyle,mksize, title, labels, xlb, ylb,stri,xlim,ylim,filename):
    ps.set_mode("small")
    fig=plt.gcf()
    #stride=2
    for line in range(numb_of_line):
        plt.plot(datas[line], color=ps.colors[line],ls=ls[line],marker = markerstyle[line], markersize=mksize, label=labels[line],markevery=stri)
    plt.legend(loc=2)
    if xlim!=[]:
        plt.xlim(xlim)
    if ylim!=[]:
        plt.ylim(ylim)
    if title!='':
        plt.title(title)
    plt.xlabel(xlb)
    plt.ylabel(ylb)
    plt.show()
    fig.tight_layout()
    name=filename+'.pdf'
    fig.savefig(name)
#####################################################################
def sub_energy(energy):
    ps.set_mode("tiny")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    ax.set_xlabel('common xlabel')
    ax.set_ylabel('common ylabel')

    axarr0=subplot(2,1,1)
    axarr0.plot(energy[0],ls='-',marker='',label='Office 1',markevery=stride)
    axarr0.plot(energy[1],ls='-',marker='+',label='Office 2',markevery=stride)
    axarr0.plot(energy[2],ls='-',marker='.',label='Office 3',markevery=stride)
    #axarr0.set_xlim([0,runTime])
    axarr0.set_ylim([0,200])
    axarr0.legend(loc=1, ncol=1, shadow=True,  fancybox=False)
    #xticks([]), yticks([0,20,40,60,80])

    axarr1=subplot(2,1,2)
    axarr1.plot(energy[3],ls='--',marker='+',label='DC 1',markevery=stride)
    axarr1.plot(energy[4],ls='--',marker='.',label='DC 2',markevery=stride)
    axarr1.plot(energy[5],ls='--',marker='',label='DC 3',markevery=stride)
    axarr1.plot(energy[6],ls='-',marker='',label='BK',markevery=stride)
    #axarr1.set_xlim([0,runTime])
    #axarr1.set_ylim([-5,300])


    axarr1.legend(loc=1, ncol=1, shadow=True,  fancybox=False)

    #xticks([]), yticks([0,20,40,60,80])

    # axarr2=subplot(3,1,3)
    # axarr2.plot(energy[6],ls='-',marker='.',label='Backup')
    #
    #
    # axarr2.legend(loc="right", ncol=1, shadow=True,  fancybox=False)
    # axarr2.set_ylim([75,80])
    # axarr2.set_xlim([0,30])
    # yticks([75,77,79])
    fig.text(0.5, 0.04, 'Iterations', ha='center', va='center')
    fig.text(0.015, 0.5, 'Energy (Kwh)', ha='center', va='center', rotation='vertical')
    plt.show()
    fig.tight_layout()
    fig.savefig('energy_conv.pdf')
#############################################################
def totalCost(DANE,baseline1,baseline2,opt):
    #
    ps.set_mode("tiny")
    fig=plt.figure()
    stride=4
    plt.plot(DANE, marker = '^', markersize=5, label='DAMESH',markevery=stride)
    plt.plot(baseline1, ls='--',marker = '+', markersize=5, label='Baseline 1',markevery=stride)
    plt.plot(baseline2, ls='-',marker = 'o', markersize=5, label='Baseline 2',markevery=stride)
    #optimal value
    plt.plot(np.ones(runTime)*opt,ls='--', markersize=5, label='Optimal')
    plt.xlim([0,100])
    #plt.ylim([0,10])
    #plt.ylim([35,60])
    plt.legend(loc=1)
    plt.xlabel('Iterations')
    plt.ylabel('Total cost')
    plt.show()
    fig.tight_layout()
    fig.savefig('plot/totalcost_conv.pdf')

#########################################################
def totalCost_trace(DANE,baseline1,baseline2):
    #

    fig=plt.figure(figsize=(6,3.5))
    stride=5
    plt.plot(DANE, marker = '^', markersize=5, label='DANE',markevery=stride)
    plt.plot(baseline1-0.3, ls='--',marker = '+', markersize=5, label='Baseline 1',markevery=stride)
    plt.plot(baseline1, ls='--',marker = 'o', markersize=5, label='Baseline 2',markevery=stride)
    #optimal value
    #plt.plot(np.ones(runTime)*opt,ls='--', markersize=5, label='Optimal')
    plt.xlim([0,50])
    #plt.ylim([0,10])
    #plt.ylim([35,60])
    plt.legend(loc=1)
    plt.xlabel('Time slots')
    plt.ylabel('Total cost')
    plt.show()

    fig.savefig('plot/totalcost_trace.pdf')
#######################################################
def costConv(cost):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    ax.set_xlabel('common xlabel')
    ax.set_ylabel('common ylabel')

    axarr0=subplot(3,1,1)
    axarr0.plot(cost[0],ls='-',marker='',label='Office 1')
    axarr0.plot(cost[1],ls='-',marker='+',label='Office 2')
    axarr0.plot(cost[2],ls='-',marker='.',label='Office 3')
    axarr0.set_xlim([0,30])

    axarr0.legend(loc="right", ncol=1, shadow=True,  fancybox=False)
    xticks([]), yticks([])

    axarr1=subplot(3,1,2)
    axarr1.plot(cost[3],ls='--',marker='+',label='DC 1')
    axarr1.plot(cost[4],ls='--',marker='.',label='DC 2')
    axarr1.plot(cost[5],ls='--',marker='',label='DC 3')
    axarr1.set_xlim([0,30])
    axarr1.set_ylim([55,70])
    axarr1.legend(loc="right", ncol=1, shadow=True,  fancybox=False)
    xticks([]), yticks([])

    axarr2=subplot(3,1,3)
    axarr2.plot(cost[6],ls='-',marker='.',color='black',label='Backup')
    axarr2.legend(loc="right", ncol=1, shadow=True,  fancybox=False)
    axarr2.set_xlim([0,30])
    yticks([75,77,79])
    fig.text(0.5, 0.04, 'Iterations', ha='center', va='center')
    fig.text(0.06, 0.5, 'Energy (Kwh)', ha='center', va='center', rotation='vertical')
    fig.savefig('energy_conv.pdf')


##############################################################

### Alpha comparison    
def plot_alpha(nu_SWO, price_convg_list_alpha, alpha_prov_pay_list, alpha_list):
    ps.set_mode("small") 
    fig, (ax1, ax2) = plt.subplots(1, 2)

    x_range = range(len(price_convg_list_alpha) + 1)    
    stride = 10.
    for i in x_range:
        if i != max(x_range):
            x = range(len(price_convg_list_alpha[i]))       
            y = price_convg_list_alpha[i]
            current_label = r"$\alpha=${}".format(alpha_list[i])
        else:
            x = range(len(nu_SWO))
            y = nu_SWO
            current_label = r'$\nu^{*}$'
#        if i < len(linestyles):
#            ax1.plot(x,y, lw=linewidth, color=ps.colors[i], alpha=0.6, label = current_label, ls = linestyles[i])
#        else:
        ax1.plot(x,y, lw=linewidth, color=ps.colors[i], alpha=0.6, label =current_label, ls = linestyles[0], marker = markers[i], markevery=stride)
      
 #   ax1.plot(x,y, lw=linewidth, color=ps.colors[i], alpha=0.6, label = current_label, marker = markers[i])
            
    y_max = max([i[-1] for i in price_convg_list_alpha])
    ax1.set_ylim(0.5*y_max, 2.5*y_max )
    ax1.set_xlabel('iterations')
    ax1.set_ylabel(r'$g(\Theta^{ne})$')
# Legend   
    leg = ax1.legend(loc="upper center", ncol=2, fancybox=True, shadow=True, bbox_to_anchor=(0.5, 1.10))
#    leg.get_frame().set_facecolor('0.9')    # set the frame face color to light gray
#    leg.get_frame().set_alpha(0.99)
#    for t in leg.get_texts():
#        t.set_fontsize('small')    # the legend text fontsize
#    for l in leg.get_lines():
#        l.set_linewidth(1.0)  # the legend line width
    
    x = alpha_list
    y = alpha_prov_pay_list
    ax2.plot(x,y, lw=linewidth, alpha=0.6)
    
    ax2.set_xlabel(r'$\alpha$')
    ax2.set_ylabel(r'$\sum\limits_{i} R_i$')
    
    fig.tight_layout()
    plt.savefig('alpha.pdf')
    plt.show() 
        
#def plot_provider_payment_alpha(alpha_prov_pay_list, alpha_list):
#    ps.set_mode("small") 
#    fig, ax = plt.subplots()
#    
#    x = alpha_list
#    y = alpha_prov_pay_list
#    ax.plot(x,y, lw=linewidth, alpha=0.6)
#    
#    ax.set_xlabel(r'$\alpha$')
#    ax.set_ylabel(r'$\sum\limits_{i} R_i$')
#    ax.legend(loc="best")
#    plt.savefig('total_payment_alpha.pdf')
#    plt.show() 
    
### Convergence    
def plot_price_convg(provider, k): 
    ps.set_mode("small") 
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)    
    
    x = np.arange(k)[start_iter:]
    y = provider.price_convg_list[start_iter:]
    ax1.plot(x,y, lw=linewidth, alpha=0.6, marker = markers[1])     
    y_max = provider.price_convg_list[-1]
    ax1.set_ylim(0.5*y_max, 1.5*y_max )   
    ax1.set_xlabel('iterations')
    ax1.set_ylabel(r'$g(\Theta^{ne})$')
   
    
    
    y = provider.price_convg_SWO_list[start_iter:]
    ax2.plot(x,y, lw=linewidth, alpha=0.6, marker = markers[2])   
    y_max = provider.price_convg_SWO_list[-1]
    ax2.set_ylim(0.5*y_max, 1.5*y_max )
    ax2.set_xlabel('iterations')
    ax2.set_ylabel(r'$\nu^*$')
    ax2.legend(loc="upper left")
    
    fig.tight_layout()
    plt.savefig('price_convg.pdf')
    plt.show() 
#def plot_price_convg_SWO(provider, k): 
#    ps.set_mode("small") 
#    fig, ax = plt.subplots()
#    x = np.arange(k)[start_iter:]
#    y = provider.price_convg_SWO_list[start_iter:]
#    ax.plot(x,y, lw=linewidth, alpha=0.6, marker = markers[1]) 
#    
#    y_max = provider.price_convg_SWO_list[-1]
#    ax.set_ylim(0., 1.5*y_max )
#    ax.set_xlabel('iterations')
#    ax.set_ylabel(r'$\nu^*$')
#    ax.legend(loc="best")
#    plt.show() 

    
def plot_bid_convg(tenant_list, k):
    ps.set_mode("small")    
    fig, ax = plt.subplots()
    x = np.arange(k)[start_iter:]
    stride = max( int(len(x)/10), 1 )
    for i in range(len(tenant_list)):
        y = tenant_list[i].bid_convg_list[start_iter:]
        current_label = r'Tenant {}'.format(i+1)
        
#        if i < len(linestyles):
#            ax.plot(x,y, lw=linewidth, color=ps.colors[i], alpha=0.6, label = current_label, ls = linestyles[i])
#        else:
        ax.plot(x,y, lw=linewidth, color=ps.colors[i], alpha=0.6, label = current_label, ls = linestyles[0], marker = markers[i], markevery=stride) 
   
    ax.set_xlabel('iterations')
    ax.set_ylabel(r'$\theta_i$')
    ax.legend(loc="upper left")
    fig.tight_layout()
    plt.savefig('bid_convg.pdf')
    plt.show()

def plot_reward_convg(tenant_list, k):
    ps.set_mode("small")    
    fig, ax = plt.subplots()
    x = np.arange(k)[start_iter:]
    stride = max( int(len(x)/10), 1 )
    for i in range(len(tenant_list)):
        y = tenant_list[i].reward_convg_list[start_iter:]
        current_label = r'Tenant {}'.format(i+1)
#        if i < len(linestyles):
#            ax.plot(x,y, lw=linewidth, color=ps.colors[i], alpha=0.6, label = current_label, ls = linestyles[i])
#        else:
        ax.plot(x,y, lw=linewidth, color=ps.colors[i], alpha=0.6, label = current_label, ls = linestyles[0], marker = markers[i], markevery=stride)
      
    y_max = max([i.reward_convg_list[-1] for i in tenant_list])
    ax.set_ylim(0., 1.5*y_max )
    ax.set_xlabel('iterations')
    ax.set_ylabel(r'$R_i$')
    ax.legend(loc="upper left")
    fig.tight_layout()
    plt.savefig('reward_convg.pdf')
    plt.show() 
    
def plot_e_convg_SWO(tenant_list, k):
    ps.set_mode("small")    
    fig, ax = plt.subplots()
    x = np.arange(k)[start_iter:]
    stride = max( int(len(x)/10), 1 )
    for i in range(len(tenant_list)):
        y = (np.array(tenant_list[i].e_convg_SWO_list))[start_iter:]*scale
        current_label = r'Tenant {}'.format(i+1)
#        if i < len(linestyles):
#            ax.plot(x,y, lw=linewidth, color=ps.colors[i], alpha=0.6, label = current_label, ls = linestyles[i])
#        else:
        ax.plot(x,y, lw=linewidth, color=ps.colors[i], alpha=0.6, label = current_label, ls = linestyles[0], marker = markers[i], markevery=stride)
    ax.set_xlabel('iterations')
    ax.set_ylabel(r'$\Delta e_i$')
    ax.legend(loc="upper left")
    fig.tight_layout()
    plt.savefig('e_convg_SWO.pdf')
    plt.show() 


### Hours comparison   
def plot_tenant_reward(tenant_reward_hours_array, num_hours, num_tenants,scheme=1): 
    ps.set_mode("small") 
    fig, ax = plt.subplots()
    
    width = 0.5
    patterns = ["|||", "\\"  ,"///" , "...", "---" , "xxx" , "+++" ,    "o",   "O",  "*" ]
    ind = np.arange(num_hours) + 1
    name = ['Tenant 1', 'Tenant 2', 'Tenant 3','Tenant 4','Tenant 5']    
    bottom = np.zeros(num_hours)
    
    for i in range(num_tenants):
        ax.bar(ind, tenant_reward_hours_array[:,i], width=width, color= 'white', align='center', edgecolor='black', \
        label=name[i], hatch=patterns[i],  bottom=bottom)
        bottom += tenant_reward_hours_array[:,i]
        
    ax.set_xlabel(r'Time (h)')
    ax.set_ylabel(r'$R_i$')
    ax.set_xlim([-0.5, num_hours + 1])
    
    if scheme == 1:
       # ax.set_title('EPM')
        plt.savefig('tenant_reward_EPM.pdf')
    elif scheme ==2:
       # ax.set_title('SWO')
        plt.savefig('tenant_reward_SWO.pdf')
    else:
        ax.legend(loc="best")
       # ax.set_title('RAND') 
        plt.savefig('tenant_reward_RAND.pdf')
    plt.show() 
def plot_tenant_e(tenant_e_hours_list, num_hours, num_tenants, scheme=1): 
    ps.set_mode("small") 
    fig, ax = plt.subplots()
    
    width = 0.5
    patterns = ["|||", "\\"  ,"///" , "...", "---" , "xxx" , "+++" ,    "o",   "O",  "*" ]
    index = np.arange(num_hours) + 1
    name = ['Tenant 1', 'Tenant 2', 'Tenant 3','Tenant 4','Tenant 5']    
    bottom = np.zeros(num_hours)
    
    for i in range(num_tenants):
        ax.bar(index, tenant_e_hours_list[:,i]*scale, width=width, color= 'white', align='center', edgecolor='black', \
        label=name[i], hatch=patterns[i],  bottom=bottom)
        bottom += tenant_e_hours_list[:,i]*scale
   
    ax.set_xlabel(r'Time (h)')
    ax.set_ylabel(r'$\Delta e_i$')
    ax.set_xlim([-0.5, num_hours+1])
        
    if scheme == 1:
       # ax.set_title('EPM')
        plt.savefig('tenant_e_EPM.pdf')
    elif scheme ==2:
       # ax.set_title('SWO')
        plt.savefig('tenant_e_SWO.pdf')
    else:
        ax.legend(loc="best")
       # ax.set_title('RAND') 
        plt.savefig('tenant_e_RAND.pdf')
    plt.show()  
    
def plot_tenant_cost(tenant_cost_hours_array, num_hours, num_tenants, scheme=1):
    ps.set_mode("small") 
    fig, ax = plt.subplots()
    
    width = 0.5
    patterns = ["|||", "\\"  ,"///" , "...", "---" , "xxx" , "+++" ,    "o",   "O",  "*" ]
    ind = np.arange(num_hours) + 1
    name = ['Tenant 1', 'Tenant 2', 'Tenant 3','Tenant 4','Tenant 5']    
    bottom = np.zeros(num_hours)
    
    for i in range(num_tenants):
        ax.bar(ind, tenant_cost_hours_array[:,i], width=width, color= 'white', align='center', edgecolor='black', \
        label=name[i], hatch=patterns[i],  bottom=bottom)
        bottom += tenant_cost_hours_array[:,i]
   
    ax.set_xlabel(r'Time (h)')
    ax.set_ylabel(r'$C_i(m_i)$')
    ax.set_xlim([-0.5, num_hours + 1])
     
    if scheme == 1:
       # ax.set_title('EPM')
        plt.savefig('tenant_cost_EPM.pdf')
    elif scheme ==2:
       # ax.set_title('SWO')
        plt.savefig('tenant_cost_SWO.pdf')
    else:
        ax.legend(loc="best")
       # ax.set_title('RAND') 
        plt.savefig('tenant_cost_RAND.pdf')
    plt.show() 
     

def plot_sum_cost(sum_cost_all_array, num_hours, num_tenants):
    ps.set_mode("small") 
    fig, ax = plt.subplots()    
    index = np.arange(num_hours)+1
    bar_width = 0.25
    space = 0.01
    opacity = 0.99
    error_config = {'ecolor': '0.3'}
    patterns = ["|||", "\\"  ,"///" , "...", "---" , "xxx" , "+++" ,    "o",   "O",  "*" ]
    
    alg1 = ax.bar(index, sum_cost_all_array[0,:]*scale, bar_width, 
                 alpha=opacity,
                 color= 'white',
                 error_kw=error_config, hatch=patterns[1],
                 label='EPM', align='center')  
    alg2 = ax.bar(index + bar_width + space, sum_cost_all_array[1,:]*scale, bar_width,
                 alpha=opacity,
                 color= 'white',
                 error_kw=error_config, hatch=patterns[2],
                 label='SWO', align='center')               
    alg3 = ax.bar(index + 2*(bar_width + space), sum_cost_all_array[2,:]*scale, bar_width,
                 alpha=opacity,
                 color= 'white',
                 error_kw=error_config, hatch=patterns[3],
                 label='RAND',align='center') 
#    autolabel(alg1)
#    autolabel(alg2, 0.01)
#    autolabel(alg3, 0.03)
    plt.xticks(index + bar_width, index, rotation='0', fontsize=8)
    ax.set_xlim(0,len(index) + 2)
    y_max = sum_cost_all_array.max()*scale
    ax.set_ylim(0., 1.1*y_max )
    ax.set_ylabel(r"$\sum\nolimits_{i} C_i(m_i)$")
    ax.set_xlabel(r'Time (h)')
    ax.legend(loc="lower right")
    fig.tight_layout()
    plt.savefig("sum_cost.pdf")
    
def plot_tenant_util(tenant_reward_hours_array, tenant_cost_hours_array, num_hours, num_tenants, scheme=1):
    ps.set_mode("small") 
    fig, ax = plt.subplots()
    
    width = 0.5
    patterns = ["|||", "\\"  ,"///" , "...", "---" , "xxx" , "+++" ,    "o",   "O",  "*" ]
    ind = np.arange(num_hours) + 1
    name = ['Tenant 1', 'Tenant 2', 'Tenant 3','Tenant 4','Tenant 5']    
    bottom = np.zeros(num_hours)
    y = tenant_reward_hours_array - tenant_cost_hours_array
    #print " Tenants' utility is: {} ".format(y)
    for i in range(num_tenants):
        ax.bar(ind, y[:,i], width=width, color= 'white', align='center', edgecolor='black', \
        label=name[i], hatch=patterns[i],  bottom=bottom)
        bottom += y[:,i]
   
    ax.set_xlabel(r'Time (h)')
    ax.set_ylabel("Tenants' utility")
    ax.set_xlim([-0.5, num_hours + 1])
     
    if scheme == 1:
       # ax.set_title('EPM')
        plt.savefig('tenant_util_EPM.pdf')
    elif scheme ==2:
       # ax.set_title('SWO')
        plt.savefig('tenant_util_SWO.pdf')
    else:
        ax.legend(loc="best")
       # ax.set_title('RAND') 
        plt.savefig('tenant_util_RAND.pdf')
    plt.show() 