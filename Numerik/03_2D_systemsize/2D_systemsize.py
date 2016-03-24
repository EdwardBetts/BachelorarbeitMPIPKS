# -*- coding: us-ascii -*-
"""
title: 2D_systemsize.py
author: Toni Ehmcke (MaNr: 3951871)
date modified: 24.03.16

Consider a 2D tight-binding lattice of Mx x My sites.
It shall be embedded in an environment with temperature T_e, coupling 
strength g_e. The site (lx,ly) is connected to a second bath with 
temperature T_h >> J and coupling strength g_h.

Objective of this program is to determine the mean field occupations n_i of
the single-particle eigenmodes i of the system.
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
import matplotlib
import matplotlib.cm as cm

def get_k(M):
    """ Return all M possible quasimomenta of a 1D tight binding chain
    in the first Brillouin zone."""
    k = np.arange(1,M+1)*np.pi/(M+1)    # vector of all possible quasimomenta
    return k

def get_vec_k(kx, ky, Mx, My):
    """ Return a vector with tuples (kx,ky) of quasimomenta."""   
    # structured data type: vector of tuples of 64-bit floats
    k = np.zeros(Mx*My, dtype=[('kx','f8'),('ky','f8')])
    for i in range(My):
        for j in range(Mx):            
            k[i*Mx + j] = (kx[j],ky[i])
    return k 
    
def get_E(k, Jx, Jy, M):
    """ Return a vector with the energy E at given quasimomenta kx, ky with the 
    dispersion relation."""
    E = np.zeros(M)
    E = -2*(Jx*np.cos(k['kx'])+Jy*np.cos(k['ky']))
    return E
    
def get_R_e(E, M, g_e, T_e):
    """ Return the contribution of the environment to the transition rate."""
    # transform energy-vector into matrices
    mat_E_x, mat_E_y = np.meshgrid(E,E) 
    mat_diff = mat_E_y - mat_E_x        # matrix representing: E_i - E_j
    # matrix for transition rates
    R_e = np.ones((M,M))* g_e**2 * T_e 
    ind = np.abs(mat_diff) > 10e-6          # indices of the non-divergent elements
    # fill in just those elements without divergences 1/0
    # the rest is set to the correct limit
    R_e[ind] = g_e**2 * mat_diff[ind]/(np.exp(mat_diff[ind]/T_e)-1)
    return R_e

def get_R_e_test(E, M, g_e, T_e, R_e, epsilon):
    """ A test routine for checking whether the calculation in get_R_e is 
    correct. Epsilon is the maximal accepted deviation between the results."""
    R_e_test = np.zeros((M,M))
    for i in range(M):
        for j in range(M):
            if np.abs(E[i]-E[j]) > 0:
                R_e_test[i,j] = g_e**2*(E[i] - E[j])/(np.exp((E[i] 
                                                        - E[j])/T_e)-1)
            else:
                R_e_test[i,j] = g_e**2 * T_e
    return np.all(np.abs(R_e_test - R_e) < epsilon)

def get_vec_sin(k, M, lx, ly):
    """ Return a vector of all possible sin**2(kx lx)*sin**2(ky*ly) terms."""
    sin = np.zeros(M)
    sin = np.sin(k['kx']*lx)**2 * np.sin(k['ky']*ly)**2
    return sin

def get_R_h(E, M, lx, ly, kx, ky, k, g_h, T_h):
    """ Return the contribution of the hot needle to the transition rate."""
    # transform energy-vector into matrices
    mat_E_x, mat_E_y = np.meshgrid(E,E)    
    mat_diff = mat_E_y - mat_E_x        # matrix representing: E_i - E_j
        
    # leave out the sine-terms at first
    # matrix for transition rates
    R_h = np.ones((M,M))*g_h**2 * T_h * 16 
    ind = np.abs(mat_diff) > 10e-6          # indices of the non-divergent elements
    # fill in just those elements without divergences 1/0
    # the rest is set to the correct limit
    R_h[ind] = g_h**2 *16* mat_diff[ind]/(np.exp(mat_diff[ind]/T_h)-1)
    
    # multiply the sine-terms
    vec_sin = get_vec_sin(k, M, lx, ly) 
    # transform sine-vectors into matrices
    mat_sin_x, mat_sin_y = np.meshgrid(vec_sin, vec_sin)
    R_h *= mat_sin_x * mat_sin_y
    
    return R_h

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
                print "Needed to repeat calculation at Temperature T_e =", T_e[-i-1],\
                        "with index i = ", N_T-i-1
                n = get_cor_n(i, T_e, r_0, M, N, E, g_e, R_h, tmpN_t, tmpN_max)
                if n == None:
                    print "Calculation failed! You may choose a larger tmpN_max."
                    break
                else:
                    r_0 = n[1:]
            else:
                r_0 = solution[0]
            mat_n[:,-i-1] = n
        if i % 10 == 0:
           print "Calculated {0:.2f}%".format(100*np.float64(i)/N_T) 
        
    print "Calculation of the occupation numbers successful!" 
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
            R_e = get_R_e(E, M, g_e,       # rate-matrix of the environment
                          tmpT[-j-1])  
            #print get_R_e_test(E, M, g_e, tmpT, R_e, 10e-15)
            R = get_R(R_e, R_h)                 # total transition rates
            data = (R, M, N)                    # arguments for fsolve  
            #-----------solve the nonlinear system of equations--------------------    
            solution = fsolve(func, r_0,args=data, full_output=1)
            if solution[2] == 0:                # if sol. didnt conv., increase s.p.
                tmpN_t *= 4
                print "Increased tmpN because of a nonconvergent solution!"
                break
            else:
                n1 = get_n1(solution[0],N) # occupation number of the ground state
                n = np.zeros(M)            # vector of all occupation numbers
                n[0], n[1:] = n1 , solution[0] 
                if np.any(n<0.):           # if solution is unphysical        
                    tmpN_t *= 4            # increase num of sample points
                    print "Increased tmpN because of an unphysical solution!"
                    break
                elif j!= tmpN_t-1:         # if iteration is not finished
                    r_0 = solution[0]      # just change initial guess 
                else:
                    print "Found a better solution! :)"
                    return n
        
        
def main():
    main.__doc__ = __doc__
    print __doc__
    
    # ------------------------------set parameters--------------------------------
    #---------------------------physical parameters--------------------------------
    Jx = 1.                             # dispersion-constant in x-direction
    Jy = 1.                             # dispersion-constant in y-direction   
    Mx = 50                             # system size in x-direction
    My = 3                             # system size in y-direction
    lx = 4.                             # heated site (x-component)
    ly = 2.                             # heated site (y-component)
    n = 3                               # particle density
    g_h = 1.                           # coupling strength needle<->system
    g_e = 1.                            # coupling strength environment<->sys
    T_h = 60*Jx                         # temperature of the needle
    M = Mx * My                         # new 2D-system size
    N = n*M                             # number of particles
    
    #----------------------------program parameters--------------------------------
    N_T = 120                           # number of temp. data-points
    tmpN_t = 4                          # number of temp. data-points in
                                        # temporary calculations
    epsilon = 10e-10                    # minimal accuracy for the compare-test
    tmpN_max = 256                      # maximal number of subslices
    T_e = np.logspace(-2,2,N_T)         # temperatures of the environment 
    r_0 = n * np.ones((M)-1)            # initial guess for n2,...,n_m
    
    #--------------------------plot parameters-------------------------------------
    n_min = 10e-3                       # minimal value of the occupation number
    n_max = N                           # maximal value of the occupation number
    nticks_cb = 5                       # number of ticks at colorbar (axK)
    T_plot = T_e[N_T/2]                 # initial env- temperature for plotting
    k_min = 0                           # lower bound for quasimomenta in plots
    k_max = np.pi                       # upper bound for quasimomenta in plots
    
    #--------------calculate environment temp. independent parameters----------
    kx = get_k(Mx)                      # vector of all quasimomenta in x-dir
    ky = get_k(My)                      # vector of all quasimomenta in y-dir
    kx_plot = kx[2]                     # initial quasimomentum in x-dir
    ky_plot = ky[0]                     # initial quasimomentum in y-dir
    k = get_vec_k(kx, ky, Mx, My)       # vector of tuples of (kx,ky)
    
    E = get_E(k, Jx, Jy, M)             # vector of all energyeigenvalues
    
    R_h = get_R_h(E, M, lx, ly, kx,     # matrix with transition rates (needle)
                  ky, k, g_h, T_h)    
    
    print "Started calculation of the occupation numbers..."    
    #-----------------------calculate occupation numbers---------------------------
    mat_n = get_mat_n(T_e, r_0, M, N_T, # matrix with occupation numbers
                      N, E, g_e, R_h,   # T_e is const. in each column
                      tmpN_t, tmpN_max) # n_i is const in each row
    
    #------------------------set - up plotting windows-----------------------------
    fig = plt.figure("Mean-field occupation", figsize=(16,14))
    
    # plotting window for n(kx,ky)
    axM = fig.add_subplot(111)
    axM.set_xlabel(r"M")
    axM.set_ylabel(r"T")
    axM.set_xlim([1,10e3])
    axM.set_ylim([10e-2,10e2])
    axM.set_xscale('log')
    axM.set_yscale('log')

    # optimize font-size   
    matplotlib.rcParams.update({'font.size': 16})
    plt.show()
              
if __name__ == '__main__':
   main()