import numpy as np

import sys
import os
sys.path.append(os.path.abspath('..'))

import time

import mf_solver as mfs
import oned_lattice as od

from matplotlib import pyplot as plt

def E(k, J):
    return 2 * J * (- np.cos(k))
    
def g(E, beta):
    if beta > 0.:
        g =  E / (np.exp(beta * E) - 1.)
    else:
        g = - E / (np.exp(beta * E) - 1.)
    g[np.isnan(g)] = 1. / np.abs(beta)
    return g

def R_generator_env(M, J, l_1s, beta_1, beta_2, gamma_1, gamma_2, retsing=False):
    x = (np.arange(M)+1)[:, np.newaxis] * np.pi / (M + 1)
    y = (np.arange(M)+1) * np.pi / (M + 1)
    R_1 = np.zeros((M, M))
    for l_1 in l_1s:
        R_1 += 1. / len(l_1s) * g(2 * J * (np.cos(y) - np.cos(x)), beta_1) * 4 * gamma_1 ** 2 * np.sin(x * l_1) ** 2 * np.sin(y * l_1) ** 2

    # get corresponding rate matrix
    R_2 = g(2 * J * (np.cos(y) - np.cos(x)), beta_2) * gamma_2 ** 2 #* np.ones((M, M))
        
    if retsing:
        return R_1, R_2
    else:
        return R_1 + R_2

# System parameters
M = 300
J = 1.
n = 3.
l = 5

gamma_h = 1.
gamma_env = 2.
Th_rel = 60
T_h =  2. * J * Th_rel
T_envs =  2. * J * np.logspace(-2, 2, 100)

#Get solutions from algorithm 1
start_time = time.time()
k = np.arange(M)*np.pi / (M+1)
r_0 = n * np.ones(M-1)
R_h = od.get_R_h(E(k, J), M, l, gamma_h, T_h)
ns_1 = od.get_mat_n(T_envs, r_0, M, n * M, E(k, J), gamma_env, (R_h), 4, 256)
print(" --- Toni: %s seconds ---" % (time.time() - start_time))

#Get solutions from algorithm 2
start_time = time.time()
R_gen = lambda x: R_generator_env(M, J, [l], 1./T_h, x, gamma_h, gamma_env)
betas, ns_2 = mfs.MF_curves_temp(R_gen, n, 1./T_envs[::-1], debug=False, usederiv=True)
print(" --- Alex: %s seconds ---" % (time.time() - start_time))

ax1 = plt.subplot(211)
ax1.loglog(T_envs[::-1], ns_2)
plt.title("Alex")
ax2 = plt.subplot(212)
ax2.loglog(T_envs, ns_1.transpose())
plt.title("Toni")

"""
diff = np.abs(ns_2[::-1] - np.transpose(ns_1))
for i in xrange(len(T_envs)):
    print "T =", T_envs[i]
    print "Differenz:", np.max(diff[i]), "in Zustand", np.argmax(diff[i])
"""
plt.show()

