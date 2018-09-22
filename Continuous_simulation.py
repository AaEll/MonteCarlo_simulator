import importlib
resources = importlib.import_module("resources")
import resources.MonteCarlo as MC

import numpy as np
import pandas as pd
import math
from functools import reduce
import random
random.seed(2010)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def cost_powers_k(rv, k = 2):
    assert(k!=1)
    if k == 2:
        cutoff = 1<<(int(rv)).bit_length()
    else:
        cutoff = k**(math.ceil(math.log(rv, k)))
    space_cost = cutoff-rv
    time_cost = (cutoff - 1)*1.0/(k-1)
    return (space_cost,time_cost,rv)

def sum_pair_costs(sim, n,verbose = True):
    sum1,sq_sum1,sum2,sq_sum2,sum3,sq_sum3 = reduce(lambda x,y : (x[0]+y[0], x[1]+y[0]**2,\
                                                     x[2]+y[1], x[3]+y[1]**2,\
                                                     x[4]+y[2], x[5]+y[2]**2),\
                                        sim.sample_repeated(n),(0,0,0,0,0,0))
    if verbose:
        print("confidence for space complexity")
        MC.Calculate_ConfIntv(sum1,sq_sum1,n,.99)
        print("confidence for time complexity")
        MC.Calculate_ConfIntv(sum2,sq_sum2,n,.99)
    return (sum1/n, sq_sum1/n, sum2/n, sq_sum2/n,sum3/n,sq_sum3/n)
    
alpha = -2
xmin = 1
n = 1000000
inv_f = lambda x : MC.inverse_continuous_power_law(x,xmin,alpha)
rv = MC.RandomVariable(inv_f,"power law, alpha = -2, for x > 1",discrete = False,verbose = False)
X  = []
Y1 = []
Y2 = []
Y3 = []

for j in range(98):
    i = 3-j/50
    print ("i = {}".format(i))
    cost_f = lambda x : cost_powers_k(x,i)
    Sim = MC.Simulator(rv,cost_f,'Live-Graph', verbose = False)
    y1, _, y2, _, y3, _ = sum_pair_costs(Sim, n, verbose = False)
    X.append(i)
    Y1.append(y1)
    Y2.append(y2)
    Y3.append(y3)
plt.plot(X,Y1,label = "space overhead")
plt.plot(X,Y2,label = "computation cost")
plt.plot(X,Y3,label = "ideal space of graph")
#plt.ylim(0,300)
# Add legend
plt.legend(loc='upper left')
# Add title and x, y labels
plt.title("Tradeoff for LiveGraph - continuous powerlaw a={}, x_min = {}".format(alpha,xmin), fontsize=16, fontweight='bold')
plt.suptitle("", fontsize=10)
plt.xlabel("Cuttoff Step Size")
plt.ylabel("Memory Used")
plt.savefig('results/continuous_fig.png')
print("finished_continuous")
