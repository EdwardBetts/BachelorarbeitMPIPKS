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
import mf_solver as mfs

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
    R_e = g_e**2 * mat_diff/(np.exp(mat_diff/T_e)-1)
    # set the divergent elements to the correct limit
    R_e[np.isnan(R_e)] = g_e**2 * T_e 
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
    R_h = g_h**2 *16* mat_diff/(np.exp(mat_diff/T_h)-1)
    # fill in just those elements without divergences 1/0
    # the rest is set to the correct limit
    R_h[np.isnan(R_h)] = g_h**2 * T_h * 16 
    
    # multiply the sine-terms
    vec_sin = get_vec_sin(k, M, lx, ly) 
    # transform sine-vectors into matrices
    mat_sin_x, mat_sin_y = np.meshgrid(vec_sin, vec_sin)
    R_h *= mat_sin_x * mat_sin_y
    return R_h

    
def get_mat_n_M(Mx, My, Jx, Jy, n, lx, ly, g_e, g_h, T_h, T_e, N_T, tmpN_t, tmpN_max, axM):
    mat_n_M = np.zeros((N_T,len(Mx)))    
    for i in range(len(Mx)):
        if i % 5 == 0:
            print "Calculated {0:.2f}%".format(100*np.float64(i)/len(Mx)) 
        M = Mx[i] * My                         # total number of sizes
        r_0 = n * np.ones(M-1)              # initial guess for n2,...,n_m
        N = n*M                             # particle number 
        print int(Mx[i])
        
        kx = get_k(int(Mx[i]))                      # vector of all quasimomenta in x-dir
        ky = get_k(My)                      # vector of all quasimomenta in y-dir
        k = get_vec_k(kx, ky, int(Mx[i]), My)       # vector of tuples of (kx,ky)
        E = get_E(k, Jx, Jy, M)             # vector of all energyeigenvalues
        R_h = get_R_h(E, M, lx, ly, kx,     # matrix with transition rates (needle)
                      ky, k, g_h, T_h)                           # particle number
        # mat_n = get_mat_n(T_e, r_0, M, N_T, # matrix with occupation numbers
        #               N, E, g_e, R_h,       # T_e is const. in each column
        #               tmpN_t, tmpN_max)     # n_i is const in each row
        R_gen = lambda x: R_h + get_R_e(E, M, g_e, 1./x)
        beta_env, ns_2 = mfs.MF_curves_temp(R_gen, n, 1./T_e[::-1], debug=False, usederiv=True)
        mat_n = np.transpose(ns_2[::-1])
        # determine index of condensate state (at T_e[0])
        ind_max = np.argmax(mat_n, axis=0)[0]
        mat_n_M[:,i] = mat_n[ind_max,:]/N
    print "Calculation finished!"
    return mat_n_M
        
        
def main():
    main.__doc__ = __doc__
    print __doc__
    
    # ------------------------------set parameters--------------------------------
    #---------------------------physical parameters--------------------------------
    Jx = 1.                             # dispersion-constant in x-direction
    Jy = 1.                             # dispersion-constant in y-direction   
    lx = 7.                             # heated site (x-component)
    ly = 1.                             # heated site (y-component)
    Mx = lx * np.logspace(1,1.9,20)+1.    # system size in x-direction
    My = 2                              # system size in y-direction
    n = 3                               # particle density
    g_h = 1.                            # coupling strength needle<->system
    g_e = 1.                            # coupling strength environment<->sys
    T_h = 60*Jx                         # temperature of the needle

    
    #----------------------------program parameters--------------------------------
    N_T = 150                           # number of temp. data-points
    tmpN_t = 4                          # number of temp. data-points in
                                        # temporary calculations
    epsilon = 10e-10                    # minimal accuracy for the compare-test
    tmpN_max = 256                      # maximal number of subslices
    T_e = np.logspace(-2,2,N_T)         # temperatures of the environment 
    n_min = 0                       # minimal value of the occupation number
    n_max = 1                           # maximal value of the occupation number
    M_min = np.min(Mx)
    M_max = np.max(Mx)
    T_min = T_e[0]
    T_max = T_e[-1]
    
    
    
    #--------------calculate environment temp. independent parameters----------
    
    
    print "Started calculation of the occupation numbers..."    
    #-----------------------calculate occupation numbers---------------------------
    #mat_n = get_mat_n(T_e, r_0, M, N_T, # matrix with occupation numbers
     #                 N, E, g_e, R_h,   # T_e is const. in each column
      #                tmpN_t, tmpN_max) # n_i is const in each row
    
    #------------------------set - up plotting windows-----------------------------
    fig = plt.figure("Mean-field occupation", figsize=(16,14))
    
    # plotting window for n(kx,ky)
    axM = fig.add_subplot(111)
    axM.set_xlabel(r"M")
    axM.set_ylabel(r"T")
    axM.set_xlim([M_min,M_max])
    axM.set_ylim([T_min,T_max])
    axM.set_xscale('log')
    axM.set_yscale('log')
    
    mat_M_n = get_mat_n_M(Mx, My, Jx, Jy, n, lx, ly, g_e, g_h, T_h, 
                          T_e, N_T, tmpN_t, tmpN_max, axM)
    norm = cm.colors.Normalize(vmax=1, vmin=0)
    graph_M = axM.imshow(mat_M_n[::-1],interpolation = 'None', cmap = cm.binary, 
               norm =norm, extent=[M_min, M_max, T_min, T_max])
    cb_axE = fig.colorbar(graph_M,ticks=[0, 0.5, 1],ax=axM, format='%.1f')

    # optimize font-size   
    matplotlib.rcParams.update({'font.size': 16})
    plt.show()
              
if __name__ == '__main__':
   main()