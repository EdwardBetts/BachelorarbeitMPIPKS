# -*- coding: us-ascii -*-
"""
title: 2D_lattice.py
author: Toni Ehmcke (MaNr: 3951871)
date modified: 21.03.16

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
import matplotlib.gridspec as gridspec
import mf_solver as mfs
import solve_equilib as seq

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

def plotE(E, k_min, k_max):
    global graph_E
    graph_E = axE.imshow(E[::-1,:], interpolation = 'None', cmap = cm.YlGnBu, 
               extent = [k_min,k_max,k_min,k_max]) 
    
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
        
def plot_axT(T_e, M, mat_n, mat_n_eq):
    """ Plots the occupation numbers for different environment temperatures."""
    for i in range(M):
        axT.plot(T_e,np.abs(mat_n[i,:]), c = 'b') 
        axT.plot(T_e,np.abs(mat_n_eq[i,:]), c = '0.75', ls=':')

def plot_axK(T_inp, T_e, N_T, mat_n, n_min, n_max, kx, ky, mat_n_eq):
    """ Draw a contourline-plot of the occupation numbers in dependency on
        the quasimomenta kx, ky for fixed enironment temperature T_inp.
        This routine also draws a vertical line at T_inp in axT."""
    global graph_K
    global vline_T
    # use global plot_mat_n for using it in other procedures
    global plot_mat_n
    global plot_mat_n_eq
    global T_plot
    # remove lines drawn before        
    if graph_K != None:    
        graph_K.remove()
        del graph_K
    if vline_T != None:
        vline_T.remove()
        del vline_T 
    # distance between T_e and the input temperature
    dist_T = np.abs(T_e - T_inp)
    # index of the element of T_e that is nearest to T_inp            
    ind_plot = np.arange(N_T)[dist_T == np.min(dist_T)][0]
    # environment temperature choosed for plotting
    T_plot = T_e[ind_plot]
    # draw a vertical line at T_plot
    vline_T = axT.axvline(T_plot, color='r')
    
    # contourmatrix of occupation numbers n(k)
    plot_mat_n = mat_n[:,ind_plot].reshape((len(ky),len(kx)))
    plot_mat_n_eq = mat_n_eq[:,ind_plot].reshape((len(ky),len(kx)))
    # create contourplot and mirror the matrix in y-direction
    graph_K = axK.imshow(plot_mat_n[::-1,:], interpolation = 'None', cmap = cm.YlOrRd,
                 norm=matplotlib.colors.LogNorm(n_min, n_max), 
                 extent = [k_min,k_max,k_min,k_max]) 

def get_BE(T_e, E, mu, k):
    """ Plots the occupation numbers for different environment temperatures
        of the Bose-Einstein-distribution in thermal equilibrium."""
    dist_E = np.abs(E - mu)
    ind_E = np.arange(len(E))[dist_E == np.min(dist_E)][0]
    E_plot = E[ind_E+1:]
    n_BE = 1./(np.exp((E_plot-mu)/T_e)-1)
    return n_BE, k[ind_E+1:]
    
    
def plot_n_k(k, kx, ky, kx_inp, ky_inp, n, n_eq, T_e, E):
    """ Draw n(kx|ky) and n(kx|ky) at given environmental temperature T_e
        and given k= (kx,ky). n ist the matrix of occupation numbers at
        fixed temperature T_e. kx is constant over each column, ky in each row.
    """
    global vline_K
    global hline_K
    global graph_kx
    global graph_ky
    # use global kx, ky for saving the clicked position
    global kx_plot
    global ky_plot
    global graph_BEx
    global graph_BEy
    # remove old graphs
    if vline_K != None:
        vline_K.remove()
        del vline_K
    if hline_K != None:
        hline_K.remove()
        del hline_K 
    if graph_kx != None:
        g = graph_kx.pop(0)
        g.remove()
        del g
    if graph_ky != None:
        g = graph_ky.pop(0)
        g.remove()
        del g
    if graph_BEx != None:
        g = graph_BEx.pop(0)
        g.remove()
        del g
    if graph_BEy != None:
        g = graph_BEy.pop(0)
        g.remove()
        del g
    # distance between all k and the input  momentum
    dist_kx = np.abs(k['kx'] - kx_inp)
    dist_ky = np.abs(k['ky'] - ky_inp)
    # index of the element of k that is nearest to k_inp            
    ind_plot_kx = np.arange(len(k['kx']))[dist_kx == np.min(dist_kx)][0]
    ind_plot_ky = np.arange(len(k['ky']))[dist_ky == np.min(dist_ky)][0]
    # momentum choosed for plotting
    kx_plot = k['kx'][ind_plot_kx]
    ky_plot = k['ky'][ind_plot_ky]
    print 'kx = %.2f, ky = %.2f'%(kx_plot,ky_plot)
    vline_K = axK.axvline(kx_plot, color='g')
    hline_K = axK.axhline(ky_plot, color='g')
    
    # occupation number at fixed ky in dependency of kx
    n_kx = n[ind_plot_ky/len(kx),:]
    n_kx_eq = n_eq[ind_plot_ky/len(kx),:]
    # occupation number at fixed kx in dependency of ky        
    n_ky = n[:,ind_plot_kx%len(kx)]
    n_ky_eq = n_eq[:,ind_plot_kx%len(kx)]
    graph_kx = axKx.plot(kx,n_kx, color='g')
    graph_ky = axKy.plot(n_ky,ky, color='g')
    
    # plot BE-distribution
    #ind_max_n = np.argmax(n)    # no reshape necessary
    #mu = E[ind_max_n]
    #mat_E = E.reshape((len(ky),len(kx)))
    # energy at fixed ky in dependency of kx
    #E_kx = mat_E[ind_plot_ky/len(kx),:]
    # energy at fixed kx in dependency of ky
    #E_ky = mat_E[:,ind_plot_kx%len(kx)]
    # BE-distribution
    #BE_x, kx_BE = get_BE(T_e, E_kx, mu, kx)
    #BE_y, ky_BE = get_BE(T_e, E_ky, mu, ky)
    graph_BEx = axKx.plot(kx, n_kx_eq, color='b')
    graph_BEy = axKy.plot(n_ky_eq, ky, color='b')
    
def onMouseClick(event):
    """ Implements the click interactions of the user.
        T_e, k_x and k_y shall be chooseable by the user."""
    mode = plt.get_current_fig_manager().toolbar.mode
    if  event.button == 1 and event.inaxes == axT and mode == '':
        # find the clicked position and draw a vertical line
        T_click = event.xdata
        plot_axK(T_click, T_e, N_T, mat_n, n_min, n_max, kx, ky, mat_n_eq)
        plot_n_k(k, kx, ky, kx_plot, ky_plot, plot_mat_n, plot_mat_n_eq, T_plot, E)
        
    elif event.button == 1 and event.inaxes == axK and mode == '':
        # find the clicked position
        kx_click = event.xdata
        ky_click = event.ydata
        plot_n_k(k, kx, ky, kx_click, ky_click, plot_mat_n, plot_mat_n_eq, T_plot, E)
    # refreshing
    fig.canvas.draw()
    
#def main():
print __doc__

# ------------------------------set parameters--------------------------------
#---------------------------physical parameters--------------------------------
Jx = 1.                             # dispersion-constant in x-direction
Jy = 1.                             # dispersion-constant in y-direction   
Mx = 20                             # system size in x-direction
My = 21                             # system size in y-direction
lx = 6.                             # heated site (x-component)
ly = 1.                             # heated site (y-component)
n = 3                               # particle density
g_h = 1.                            # coupling strength needle<->system
g_e = 1.                            # coupling strength environment<->sys
T_h = 60*Jx                         # temperature of the needle
M = Mx * My                         # new 2D-system size
N = n*M                             # number of particles

#----------------------------program parameters--------------------------------
N_T = 100                           # number of temp. data-points
T_min = 1e-2                        # minimal temperature
T_max = 1e2                         # maximal temperature
T_e = np.logspace(-2,2,N_T)         # temperatures of the environment 

#--------------------------plot parameters-------------------------------------
graph_K = None                      # initialise graph at axK
vline_T = None                      # initialise vertical line at axT
vline_K = None                      # initialise vertical line at axK
hline_K = None                      # initialise horizontal line at axK
plot_mat_n = None                   # occupation number at fixed T_e
plot_mat_n_eq = None                # BE-occupation number at fixed T_e
graph_kx = None                     # initialise graph at axKx
graph_ky = None                     # initialise graph at axKy
graph_BEx = None                    # initialise graph for BEx at axKx
graph_BEy = None                    # initialise graph for BEy at axKy
graph_E = None
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
if np.abs(g_h) >= 10e-10:
    R_gen = lambda x: R_h + get_R_e(E, M, g_e, 1./x)
    beta_env, ns_2 = mfs.MF_curves_temp(R_gen, n, 1./T_e[::-1], debug=True, usederiv=True)
    mat_n = np.transpose(ns_2[::-1])
    mat_n_eq = seq.get_eq_mfo(Jx, Jy, Mx, My, lx, ly, n, g_e, T_h, N_T, T_min, T_max)
else:
    mat_n = seq.get_eq_mfo(Jx, Jy, Mx, My, lx, ly, n, g_e, T_h, N_T, T_min, T_max)
    mat_n_eq = mat_n

#------------------------set - up plotting windows-----------------------------
fig = plt.figure("Mean-field occupation", figsize=(16,14))
gs = gridspec.GridSpec(2, 2)        # grid spect for controlling figures

axK = fig.add_subplot(gs[0,0])
axK.set_xlabel(r"$k_x$")
axK.set_ylabel(r"$k_y$")
axK.set_xlim([0,k_max])
axK.set_ylim([k_min,k_max])

# plotting window for n(kx|ky)
axKx = fig.add_subplot(gs[1,0])
axKx.set_xlabel(r"$k_x$")
axKx.set_ylabel(r'$\bar{n}_i$')
axKx.set_xlim([k_min,k_max])
axKx.set_ylim([n_min,n_max])
axKx.set_yscale('log')

# plotting window for n(ky|kx)
axKy = fig.add_subplot(gs[0,1])
axKy.set_xlabel(r'$\bar{n}_i$')
axKy.set_ylabel(r"$k_y$")
axKy.set_xlim([n_min,n_max])
axKy.set_xscale('log')
axKy.set_ylim([k_min,k_max])

# plotting window for n(T_e)
axT = fig.add_subplot(gs[1,1])
axT.set_xlabel(r'$T/J$')
axT.set_ylabel(r'$\bar{n}_i$')
axT.set_xlim([np.min(T_e), np.max(T_e)])
axT.set_ylim([n_min, n_max])
axT.set_xscale('log')
axT.set_yscale('log')

# initial plots on program start
plot_axT(T_e, M, mat_n, mat_n_eq)
plot_axK(T_plot, T_e, N_T, mat_n, n_min, n_max, kx, ky, mat_n_eq)
plot_n_k(k, kx, ky, kx_plot, ky_plot, plot_mat_n, plot_mat_n_eq, T_plot, E)

# set-up-colorbar at axK
t_K = np.logspace(np.log10(n_min),np.log10(n_max), num=nticks_cb)
cb_axK = fig.colorbar(graph_K, ax=axK, ticks=t_K, format='$%.1e$')

# connect plotting window with the onClick method
cid = fig.canvas.mpl_connect('button_press_event', onMouseClick) 
# optimize font-size   
matplotlib.rcParams.update({'font.size': 15})
plt.show()
              
#if __name__ == '__main__':
#   main()