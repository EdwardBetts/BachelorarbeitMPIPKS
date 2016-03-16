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
    R_e = np.ones((M,M))*g_e**2 * T_e/4 # matrix for transition rates
    ind = np.abs(mat_diff) > 0          # indices of the non-divergent elements
    # fill in just those elements without divergences 1/0
    # the rest is set to the correct limit
    R_e[ind] = g_e**2 * mat_diff[ind]/(4*np.exp(mat_diff[ind]/T_e)-4)
    return R_e

def get_R_e_test(E, M, g_e, T_e, R_e, epsilon):
    """ A test routine for checking whether the calculation in get_R_e is 
    correct. Epsilon is the maximal accepted deviation between the results."""
    R_e_test = np.zeros((M,M))
    for i in range(M):
        for j in range(M):
            if i != j:
                R_e_test[i,j] = g_e**2*(E[i] - E[j])/(4*np.exp((E[i] 
                                                        - E[j])/T_e)-4)
            else:
                R_e_test[i,j] = g_e**2 * T_e/4
    return np.abs(R_e_test - R_e) < epsilon

def get_R_h(E, M, l, g_h, T_h):
    """ Return the contribution of the hot needle to the transition rate."""
    # transform energy-vector into matrices
    mat_E_x, mat_E_y = np.meshgrid(E,E)    
    mat_diff = mat_E_y - mat_E_x        # matrix representing: E_i - E_j
        
    # leave out the sine-terms at first
    R_h = np.ones((M,M))*g_h**2 * T_h   # matrix for transition rates
    ind = np.abs(mat_diff) > 0          # indices of the non-divergent elements
    # fill in just those elements without divergences 1/0
    # the rest is set to the correct limit
    R_h[ind] = g_h**2 * mat_diff[ind]/(np.exp(mat_diff[ind]/T_h)-1)
    
    # multiply the sine-terms
    sin = np.sin(l*np.arange(1,M+1))**2 # vector with sine-values sin(li)**2
    # transform sine-vectors into matrices
    mat_sin_x, mat_sin_y = np.meshgrid(sin,sin)
    R_h *= mat_sin_x * mat_sin_y
    
    return R_h

def get_R_h_test(E, M, l, g_h, T_h, R_h, epsilon):
    """ A test routine for checking whether the calculation in get_R_h is 
    correct. Epsilon is the maximal accepted deviation between the results."""
    R_h_test = np.zeros((M,M))
    sin = np.sin(l*np.arange(1,M+1))**2
    for i in range(M):
        for j in range(M):
            if i != j:
                R_h_test[i,j] = g_h**2*(E[i] - E[j]) * sin[i] * sin[j]\
                            /(np.exp((E[i] - E[j])/T_h)-1)
            else:
                R_h_test[i,j] = g_h**2 * T_h * sin[i] * sin[j]
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
                print i
                n = get_tmp_n(i, T_e, r_0, M, N, E, g_e, R_h, tmpN_t, tmpN_max)
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

def get_tmp_n(i, T_e, r_0, M, N, E, g_e, R_h, tmpN_t, tmpN_max):
    """ Solve the nonlinear system of equations in dependency of the 
        environment temperature T_e and return a matrix of occupation
        numbers n_i(T).
        Important for the numerics is the initial guess r_0 of the 
        occupation distribution (n_2,..,n_M) for the highest temperature 
        T_e[-1]. """
    while tmpN_t < tmpN_max:                     # repeat until tmpN_t reaches max.
        tmpT = get_tmpT(T_e, i, tmpN_t)         # reduced temperature array
                                                # for closer sample points    
        for j in range(tmpN_t):
            R_e = get_R_e(E, M, g_e, tmpT[-j-1])# matrix with transition rates (env)
            #print get_R_e_test(E, M, g_e, tmpT, R_e, 10e-15)
            R = get_R(R_e, R_h)                 # total transition rates
            data = (R, M, N)                    # arguments for fsolve  
            #-----------solve the nonlinear system of equations--------------------    
            solution = fsolve(func, r_0,args=data, full_output=1)
            if solution[2] == 0:         # if sol. didnt conv., increase s.p.
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
    print "Calculation failed!"
        

def main():
    main.__doc__ = __doc__
    print __doc__
    
    # ------------------------------set parameters-----------------------------
    J = 1.                              # dispersion-constant
    M = 300                             # system size (number of sites)
    n = 3                               # density
    N = n*M                             # number of particles
    l = 5                               # heated site
    g_h = 1.                            # coupling strength needle<->system
    g_e = 1.                            # coupling strength environment<->sys
    T_h = 60 * J                        # temperature of the needle
    N_T = 100                           # number of temp. data-points
    tmpN_t = 4                          # number of temp. data-points in
                                        # temporary calculations
    tmpN_max = 256                      # maximal number of subslices
    T_e = np.logspace(-2,2,N_T)         # temperatures of the environment    
    
    #--------------calculate environment temp. independent parameters----------
    k = get_k(M)                        # vector of all quasimomenta
    E = get_E(J,k)                      # vector of all energyvalues
    R_h = get_R_h(E, M, l, g_h, T_h)    # matrix with transition rates (needle)
    #print get_R_h_test(E, M, l, g_h, T_h, R_h, 10e-15)
    
    #-----calculate the occupation numbers in dependency of the temp-----------
    r_0 = n * np.ones(M-1)              # initial guess for n2,...,n_m
    mat_n = get_mat_n(T_e, r_0, M, N_T, N, E, g_e, R_h, tmpN_t, tmpN_max)            
    
    # print mat_n
    #-----------------------plot n_i(T_E)--------------------------------------
    fig = plt.figure("Mean-field occupation", figsize=(15,8))
    axOcc = fig.add_subplot(111)
    axOcc.set_xlabel(r'$T/J$')
    axOcc.set_ylabel(r'$\bar{n}_i$')
    axOcc.set_xlim([np.min(T_e), np.max(T_e)])
    axOcc.set_ylim([8*10e-3, 3*N])
    axOcc.set_xscale('log')
    axOcc.set_yscale('log')
    axOcc.set_title('Mean-field occupation') 
    matplotlib.rcParams.update({'font.size': 20})
    
    for i in range(M):
        axOcc.plot(T_e,np.abs(mat_n[i,:]), c = 'b')
        
    plt.show()
              
if __name__ == '__main__':
    main()