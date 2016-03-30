# -*- coding: us-ascii -*-
"""
title: 1D_latice.py
author: Toni Ehmcke (MaNr: 3951871)
date modified: 15.03.16

Consider a 1D tight-binding chain of M sites 1,..,M.
The dispersion-relation is E(k) = -2Jcos(k).
It shall be embedded in an environment with temperature T_e, coupling 
strength g_e. The site l << M is connected to a second bath with 
temperature T_h >> J.

Objective of this program is to determine the mean field occupations n_i of
the single-particle eigenmodes i of the system.
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
import matplotlib
import mf_solver as mfs

def get_k(M):
    """ Return all M possible quasimomenta of the tight binding chain
    in the first Brillouin zone."""
    k = np.arange(1,M+1)*np.pi/(M+1)    # vector of all possible quasimomenta
    return k
    
def get_E(J,k):
    """ Return the energy E at given quasimomentum k with the 
    dispersion relation. """
    E = -2 * J * np.cos(k)              # energyeigenvalue 
    return E
    
def get_R_e(E, M, g_e, T_e):
    """ Return the contribution of the environment to the transition rate."""
    # transform energy-vector into matrices
    mat_E_x, mat_E_y = np.meshgrid(E,E) 
    mat_diff = mat_E_y - mat_E_x        # matrix representing: E_i - E_j
    # matrix for transition rates
    R_e = g_e**2 * mat_diff/(np.exp(mat_diff/T_e)-1)
    R_e[np.isnan(R_e)] = g_e**2 * T_e      
    # indices of the non-divergent elements
    # fill in just those elements without divergences 1/0
    # the rest is set to the correct limit
   
    return R_e

def get_R_e_test(E, M, g_e, T_e, R_e, epsilon):
    """ A test routine for checking whether the calculation in get_R_e is 
    correct. Epsilon is the maximal accepted deviation between the results."""
    R_e_test = np.zeros((M,M))
    for i in range(M):
        for j in range(M):
            if i != j:
                R_e_test[i,j] = g_e**2*(E[i] - E[j])/(np.exp((E[i] 
                                                        - E[j])/T_e)-1)
            else:
                R_e_test[i,j] = g_e**2 * T_e
    return np.abs(R_e_test - R_e) < epsilon

def get_R_h(E, M, l, g_h, T_h):
    """ Return the contribution of the hot needle to the transition rate."""
    # transform energy-vector into matrices
    mat_E_x, mat_E_y = np.meshgrid(E,E)    
    mat_diff = mat_E_y - mat_E_x        # matrix representing: E_i - E_j
        
    # leave out the sine-terms at first
    R_h = g_h**2 *4* mat_diff/(np.exp(mat_diff/T_h)-1)    
    R_h[np.isnan(R_h)] = g_h**2 * T_h*4   # matrix for transition rates
    # fill in just those elements without divergences 1/0
    # the rest is set to the correct limit

    # multiply the sine-terms
    sin = np.sin(l*np.arange(1,M+1)*np.pi/(M+1))**2 # vector with sine-values sin(li)**2
    # transform sine-vectors into matrices
    mat_sin_x, mat_sin_y = np.meshgrid(sin,sin)
    R_h *= mat_sin_x * mat_sin_y
    
    return R_h

def get_R_h_test(E, M, l, g_h, T_h, R_h, epsilon):
    """ A test routine for checking whether the calculation in get_R_h is 
    correct. Epsilon is the maximal accepted deviation between the results."""
    R_h_test = np.zeros((M,M))
    sin = np.sin(l*np.arange(1,M+1)*np.pi/(M+1))**2
    for i in range(M):
        for j in range(M):
            if i != j:
                R_h_test[i,j] = g_h**2* 4* (E[i] - E[j]) * sin[i] * sin[j]\
                            /(np.exp((E[i] - E[j])/T_h)-1)
            else:
                R_h_test[i,j] = g_h**2 *4* T_h * sin[i] * sin[j]
    return np.abs(R_h_test - R_h) < epsilon

def get_R(R_e,R_h):
    """ Calculate the transition rate matrix. The element (i,j) accords to
    the number of transitions per time unit from energystate j to i."""
    R = R_e + R_h
    return R
    

def get_n1(r,N):
    """ Return the mean occupation number of the ground state with the
    given total particle number. r is the vector of all other mean occupation
    numbers of the energystates 2,...,M."""
    n1 = N - np.sum(r)
    return n1
    

def func(r, *data):
    """This is the function which roots should be determined.r is the vector 
    of the mean occupation numbers of the energystates 2,...,M. R is the
    matrix of transition rates. r represents the solution of the M-1 
    dimensional system of nonlinear equations."""
    R, M, N = data              # trans. rates, system size, particle number
    n1 = get_n1(r,N)            # occupation number of groundstate
    n = np.zeros(M)             # vector of mean occupation numbers
    n[0] = n1   
    n[1:] = r                    
    func = np.zeros(M)          # implement all M equations at first
    A = R - np.transpose(R)     # rate asymmetry matrix
    func = np.dot(A,n)*n + np.dot(R,n) - R.sum(axis=0) * n
    
    return func[1:]             # slice away the last equation

def get_mat_n(T_e, r_0, M, N_T, N, E, g_e, R_h, tmpN_t, tmpN_max):
    """ Solve the nonlinear system of equations in dependency of the 
        environment temperature T_e and return a matrix of occupation
        numbers n_i(T).
        Important for the numerics is the initial guess r_0 of the 
        occupation distribution (n_2,..,n_M) for the highest temperature 
        T_e[-1]. """
    mat_n = np.zeros((M,N_T))               # matrix for the result
    for i in range(N_T):
        R_e = get_R_e(E, M, g_e, T_e[-i-1]) # matrix with transition rates (env)
        #print get_R_e_test(E, M, g_e, T_e, R_e, 10e-15)
        R = get_R(R_e, R_h)                 # total transition rates
        data = (R, M, N)                    # arguments for fsolve  
        #-----------solve the nonlinear system of equations--------------------    
        solution = fsolve(func, r_0,args=data, full_output=1)
        if solution[2] == 0:                # if sol. didnt conv., repeat calcul.
            print i
        else:
            n1 = get_n1(solution[0],N) # occupation number of the ground state
            n = np.zeros(M)                 # vector of all occupation numbers
            n[0], n[1:] = n1 , solution[0] 
            if np.any(n<0.):                # if solution is unphysical      
                print "Needed to repeat calculation at Temperature T_e =", T_e[-i-1] 
                n = get_cor_n(i, T_e, r_0, M, N, E, g_e, R_h, tmpN_t, tmpN_max)
                if n == None:
                    print "Calculation failed! You may choose a larger tmpN_max."
                    break
                else:
                    r_0 = n[1:]
            else:
                r_0 = solution[0]
            mat_n[:,-i-1] = n
    return mat_n

def get_tmpT(T_e, i, tmpN_t):
    """ Calculate an array of logspaced temperature sample points with
        elements in [T_e[-i-1],T_e[-i]]."""
    if i == 0:
        print "Solution converged into an unphysical state. ",\
                "Choose a better initial guess r_0."
    else:
        tmpT = np.logspace(np.log10(T_e[-i-1]), np.log10(T_e[-i]), num= tmpN_t,
                           endpoint = True)
        return tmpT

def get_cor_n(i, T_e, r_0, M, N, E, g_e, R_h, tmpN_t, tmpN_max):
    """ If the calculation in get_mat_n throws an invalid value for a 
        particular temperature T_e[i], this function tries to repeat the 
        calculation with a smaller stepsize in the interval T_e[i-1],T_e[i]
        for getting a better initial guess.
        In the case of success it returns the correct occupation number 
        n(T_e[i]). In the case of failure (if tmpN_T >= tmpN_max) it 
        returns an exception-string."""
    while tmpN_t < tmpN_max:                    # repeat until tmpN_t reaches max.
        tmpT = get_tmpT(T_e, i, tmpN_t)         # reduced temperature array
                                                # for closer sample points    
        for j in range(tmpN_t):
            R_e = get_R_e(E, M, g_e, tmpT[-j-1])# matrix with transition rates (env)
            #print get_R_e_test(E, M, g_e, tmpT, R_e, 10e-15)
            R = get_R(R_e, R_h)                 # total transition rates
            data = (R, M, N)                    # arguments for fsolve  
            #-----------solve the nonlinear system of equations--------------------    
            solution = fsolve(func, r_0,args=data, full_output=1)
            if solution[2] == 0:                # if sol. didnt conv., increase s.p.
                tmpN_t *= 10
                break
            else:
                n1 = get_n1(solution[0],N) # occupation number of the ground state
                n = np.zeros(M)            # vector of all occupation numbers
                n[0], n[1:] = n1 , solution[0] 
                if np.any(n<0.):           # if solution is unphysical        
                    tmpN_t *= 2           # increase num of sample points
                    break
                elif j!= tmpN_t-1:         # if iteration is not finished
                    r_0 = solution[0]      # just change initial guess 
                else:
                    return n
        print "Increased tmpN!"

def plot_axT(T_e, M, mat_n):
    """ Plots the occupation numbers for different environment temperatures."""
    for i in range(M):
        axT.plot(T_e,np.abs(mat_n[i,:]), c = 'b')  
        
def plot_init_axK(T_c, T_e, E, k, N_T, mat_n):
    """Plot """
    global graph_K
    global graph_BE
    global vline
    dist_T = np.abs(T_e - T_c)
    ind_plot = np.arange(N_T)[dist_T == np.min(dist_T)][0]
    T_plot = T_e[ind_plot]
    vline = axT.axvline(T_plot, color='r')
    vline_T_c = axT.axvline(T_c, color='g')      # static line at T_c
    graph_K = axK.plot(k, mat_n[:,ind_plot], color='r')
    graph_BE = axK.plot(k[1:], get_BE(T_plot, E, k), color='b')
    
def get_BE(T_e, E, k):
    """ Plots the occupation numbers for different environment temperatures."""
    n_BE = 1./(np.exp((E[1:]-E[0])/T_e)-1)
    return n_BE

def onMouseClick(event):
    mode = plt.get_current_fig_manager().toolbar.mode
    if  event.button == 1 and event.inaxes == axT and mode == '':
        global graph_K
        global graph_BE
        global vline
        # remove lines drawn before        
        g = graph_K.pop(0)
        g.remove()
        del g
        g = graph_BE.pop(0)
        g.remove()
        del g
        vline.remove()
        del vline    
        
        # find the clicked position and draw a vertical line
        T_click = event.xdata
        dist_T = np.abs(T_e - T_click)
        ind_click = np.arange(N_T)[dist_T == np.min(dist_T)][0]
        T_plot = T_e[ind_click]
        vline = axT.axvline(T_plot, color='r')
        
        # plot n(k)
        #print mat_n[:,ind_click]
        graph_K = axK.plot(k, mat_n[:,ind_click], color='r')
        graph_BE = axK.plot(k[1:], get_BE(T_plot, E, k), color='b')
        # refreshing
        fig.canvas.draw()
        

#def main():
print __doc__

# ------------------------------set parameters-----------------------------
#---------------------------physical parameters--------------------------------
J = 1.                              # dispersion-constant
M = 202                             # system size (number of sites)
n = 3                               # density
N = n*M                             # number of particles
l = 7.                               # heated site
g_h = 1.                            # coupling strength needle<->system
g_e = 1.                            # coupling strength environment<->sys
T_h = 60 * J                        # temperature of the needle

#----------------------------program parameters--------------------------------
N_T = 100                           # number of temp. data-points
tmpN_t = 4                          # number of temp. data-points in
                                    # temporary calculations
tmpN_max = 256                      # maximal number of subslices
T_e = np.logspace(-2,2,N_T)         # temperatures of the environment 
T_c = 2 * J * n                     # critical temperature for BE-Cond.  

#---------------------------global objects for plots---------------------------
vline = None
graph_K = None
graph_BE = None
    
#--------------calculate environment temp. independent parameters----------
k = get_k(M)                        # vector of all quasimomenta
E = get_E(J,k)                      # vector of all energyvalues
R_h = get_R_h(E, M, l, g_h, T_h)    # matrix with transition rates (needle)
print np.all(get_R_h_test(E, M, l, g_h, T_h, R_h, 10e-10))
    
#-----calculate the occupation numbers in dependency of the temp-----------
#r_0 = n * np.ones(M-1)              # initial guess for n2,...,n_m
#mat_n = get_mat_n(T_e, r_0, M, N_T, N, E, g_e, R_h, tmpN_t, tmpN_max)  

R_gen = lambda x: R_h + get_R_e(E, M, g_e, 1/x)
beta_env, ns_2 = mfs.MF_curves_temp(R_gen, n, 1./T_e[::-1], debug=False, usederiv=True)       
mat_n = np.transpose(ns_2[::-1])

#-----------------------plot n_i(T_E)--------------------------------------

fig = plt.figure("Mean-field occupation", figsize=(16,9))
axT = fig.add_subplot(121)
axT.set_xlabel(r'$T/J$')
axT.set_ylabel(r'$\bar{n}_i$')
axT.set_xlim([np.min(T_e), np.max(T_e)])
axT.set_ylim([8*10e-5, 3*N])
axT.set_xscale('log')
axT.set_yscale('log')
    
axK = fig.add_subplot(122)
axK.set_xlabel(r'$k/a$')
axK.set_ylabel(r'$\bar{n}_i$')
axK.set_xlim([0, np.pi])
axK.set_ylim([8*10e-4, 3*N])
axK.set_yscale('log')

# connect plotting window with the onClick method
cid = fig.canvas.mpl_connect('button_press_event', onMouseClick)

# plot initial lines    
plot_axT(T_e, M, mat_n)
plot_init_axK(T_c, T_e, E, k, N_T, mat_n)
# refreshing
fig.canvas.draw()
    
matplotlib.rcParams.update({'font.size': 18})
plt.show()
              
#if __name__ == '__main__':
#   main()