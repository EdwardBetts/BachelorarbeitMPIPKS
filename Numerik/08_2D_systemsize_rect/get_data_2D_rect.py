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
import time
import sys

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
    
# function to save log messages to specified log file
def write_log(msg, fname):
  # open the specified log file
  file = open(fname,"a")
  # write log message with timestamp to log file
  file.write("%s: %s\n" % (time.strftime("%d.%m.%Y %H:%M:%S"), msg))
  # close log file
  file.close

def clear_log(fname):
  file = open(fname,"w")
  # close log file
  file.close

def get_mat_n_M(Mx, My, Jx, Jy, n, lx, ly, g_e, g_h, T_h, T_e, N_T, fnames):
    """ Calculate the occupation numbers in dependency of the system size M(x)
        and the environmental-temperature.
        Returns a matrix where M is constant in each column and 
        T_e is constant in each row."""
    mat_n_M = np.zeros((N_T + 2,len(Mx)))
    clear_log(fnames[3])
    write_log("Started Calculation of " + str(len(Mx)) + " elements", fnames[3])
    for i in range(len(Mx)):
        l = get_cond_cand(lx, ly, Mx[i], My[i])# determine condensate-candidates
        M = Mx[i] * My[i]                      # total number of sizes
        r_0 = n * np.ones(M-1)              # initial guess for n2,...,n_m
        N = n*M                             # particle number         
        kx = get_k(Mx[i])                   # vector of all quasimomenta in x-dir
        ky = get_k(My[i])                      # vector of all quasimomenta in y-dir
        k = get_vec_k(kx, ky, Mx[i], My[i])    # vector of tuples of (kx,ky)
        E = get_E(k, Jx, Jy, M)             # vector of all energyeigenvalues
        R_h = get_R_h(E, M, lx, ly, kx,     # matrix with transition rates (needle)
                      ky, k, g_h, T_h)
        # sum of all transition-rates in dependency of the temperature
        R_gen = lambda x: R_h + get_R_e(E, M, g_e, 1./x)
        beta_env, ns_2, log = mfs.MF_curves_temp(R_gen, n, 1./T_e[::-1], debug=False, usederiv=True)
        # matrix with occupation numbers in depedency of the temperature
        mat_n = np.transpose(ns_2[::-1])
        # determine index of condensate state (at T_e[1])
        ind_max = np.argmax(mat_n, axis=0)[1]
        mat_n_M[2:,i] = mat_n[ind_max,:]/N
        
        # determine the corresponding condensate-candidate
        diff_x = np.abs(l['lx'] - k[ind_max]['kx'])
        diff_y = np.abs(l['ly'] - k[ind_max]['ky'])
        # index of the condensate state
        ind_cond_x = np.argmin(diff_x, axis=1)[0]
        ind_cond_y = np.argmin(diff_y, axis=0)[0]
        mat_n_M[0,i] = ind_cond_x
        mat_n_M[1,i] = ind_cond_y
        mat_n_M[:,:i+1].tofile(fnames[0],sep=';')
        Mx[:i+1].tofile(fnames[1],sep=';')
        My[:i+1].tofile(fnames[2],sep=';')
        #if i % 5 == 0:
        print "Calculated {0:.2f}%".format(100*np.float64(i)/len(Mx))
        write_log("Calculated Element " + str(i) + ". " + log, fnames[3])
    print "Calculation finished!"
    write_log("Calculation finished!", fnames[3])
    return mat_n_M

def get_cond_cand(lx, ly, Mx, My):
    """ Returns an array of candidate-states for the condensate."""    
    # structured data type: vector of tuples of 64-bit floats
    l = np.zeros(((ly+1),(lx+1)), dtype=[('lx','f8'),('ly','f8')])
    for i in range(lx+1):
        for j in range(ly+1): 
            if i == 0 and j == 0:
                l[0,0] = (np.pi / (Mx + 1), np.pi / (My + 1))
            elif i == 0 and j != 0:
                l[j,i] = (np.pi / (Mx + 1), j * np.pi / ly)
            elif i != 0 and j == 0:
                l[j,i] = (i * np.pi / lx, np.pi / (My + 1))
            else:
                l[j,i] = (i * np.pi / lx, j * np.pi / ly)
    return l
        
        
def main():
    main.__doc__ = __doc__
    print __doc__
    
    # ------------------------------set parameters-----------------------------
    #----------------------------program parameters----------------------------
    fname_mat_M_n = 'mat_M_n.dat'       # file-name of the file for mat_M_n
    fname_Mx = 'Mx.dat'                 # file-name of the file for Mx
    fname_My = 'My.dat'                 # file-name of the file for Mx
    fname_log = 'log.dat'               # log-file
    fnames = [fname_mat_M_n, fname_Mx,  # array with filenames
              fname_My, fname_log]
    fname_T_e = 'T_e.dat'               # file-name of the file for T_e
    fname_params = 'params.dat'         # file-name of the file for params
    
    #N_M = 3                            # number of system-size data-points
    Mx_min = 10                         # minimal system size
    Mx_max = 15                         # maximal system size
    My_min = Mx_min-1                   # minimal system size (magnitude)
    My_max = Mx_max-1                   # maximal system size (magnitude)
    N_T = 100                           # number of temp. data-points
    T_e_min = 1e-2                      # minimal temperature
    T_e_max = 1e2                       # maximal temperature
    
    #---------------------------physical parameters----------------------------
    Jx = 1.                             # dispersion-constant in x-direction
    Jy = 1.                             # dispersion-constant in y-direction   
    lx = 4                              # heated site (x-component)
    ly = 3                              # heated site (y-component)
    n = 3                               # particle density
    g_h = 1.                            # coupling strength needle<->system
    g_e = 1.                            # coupling strength environment<->sys
    T_h = 60*Jx                         # temperature of the needle

    # vector with all parameters
    params = np.array([Jx, Jy, lx, ly, n, g_h, g_e, T_h])
    params.tofile(fname_params,sep=';')
    
    #--------------------------physical variables------------------------------
    # system size in x-direction
    Mx = np.arange(Mx_min, Mx_max+1)
    # system size in y-direction
    My = np.arange(My_min, My_max+1)
    assert len(Mx) == len(My)
    # temperatures of the environment
    T_e = np.logspace(np.log10(T_e_min),np.log10(T_e_max),N_T)  
    T_e.tofile(fname_T_e,sep=';')

    #-------------------calculate the occupation numbers-----------------------
    print "Started calculation..."
    mat_M_n = get_mat_n_M(Mx, My, Jx, Jy, n, lx, ly, g_e, g_h, T_h, 
                          T_e, N_T, fnames)
                          
    #------------------------Save all data to files----------------------------
    try:
        mat_M_n.tofile(fname_mat_M_n,sep=';')
        Mx.tofile(fname_Mx,sep=';')
        My.tofile(fname_My,sep=';')
        print "Saving of data successful!"
    except:
        print "An error occured while saving the data!"
    
                 
if __name__ == '__main__':
   main()