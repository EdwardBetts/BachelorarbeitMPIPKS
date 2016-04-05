# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 14:52:30 2016

@author: toni
"""
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec

def getParams(fname):
    """ Load file fname with saved parameters and return them."""
    params = np.fromfile(fname, sep=';')
    return  params[0], params[1], params[2], params[3], params[4],\
            params[5], params[6], params[7], params[8]

def get_conn_parts(a):
    """ Return slice-operands that choose connected parts from array a
        that are neighboured and equal.
        E.g. get_conn_parts(np.array[1,1,1,2,2,3]) will return
            [slice(0,3,None),slice(3,5,None), slice(5,6,None)]"""
    N = len(a)
    ind = 0
    slices = np.zeros(0)
    while ind < N:
        start = ind
        uneq_ind = a[ind:][0] != a[ind:]
        if np.any(uneq_ind):
            inc = np.arange(len(a[ind:]))[uneq_ind][0]
        else:
            inc = len(a[ind:])
        stop = inc + ind
        slices = np.append(slices, slice(start,stop))
        ind += inc
    return slices

def get_ax_size(ax):
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width *= fig.dpi
    height *= fig.dpi
    return width, height

def main():
    #-------------------------Load saved data----------------------------------
    fname_mat_M_n = 'mat_M_n.dat'     # file-name of the file for mat_M_n
    fname_Mx = 'Mx.dat'               # file-name of the file for Mx
    fname_T_e = 'T_e.dat'             # file-name of the file for T_e
    fname_params = 'params.dat'       # file-name of the file for params
    
    # physical parameters
    Jx, Jy, lx, ly, n, g_h, g_e, T_h, My = getParams(fname_params)   
    
    # environment temperatures
    T_e = np.fromfile(fname_T_e, sep=';')
    N_T = len(T_e)
    # system sizes
    Mx = np.fromfile(fname_Mx, dtype=int, sep=';')
    N_M = len(Mx)
    # matrix with condensate states-occupations
    mat_M_n = np.fromfile(fname_mat_M_n, sep=';').reshape(N_T+2,N_M)
    # indices of condensate states
    ix = mat_M_n[0,:]
    iy = mat_M_n[1,:]
    # index of the condensate state using cantors coupling-function
    ind_cond = (iy + (ix+iy)*(ix+iy+1)/2.).astype(int)
    print Mx
    print ind_cond
    # graphs for imshow - rowwise mirrored. Represent the fraction of the
    # condensed particles to the total particle number
    cond_frac = mat_M_n[2::][::-1]

    #------------------------Plot Parameters-----------------------------------
    n_min = 0                         # minimal value of the occupation number
    n_max = 1                         # maximal value of the occupation number
    M_min = Mx[0]                     # minmal system size
    M_max = Mx[-1]                    # maximal system size
    dM = np.float(Mx[1])/Mx[0]        # 'distance' between nighboured sizes 
    T_min = T_e[0]                    # minimal temperature
    T_max = T_e[-1]                   # maximal temperature
    cmaps = ['BuPu', 'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd','YlOrBr']
    #------------------------set - up plotting windows-------------------------
    fig = plt.figure("Mean-field occupation", figsize=(16,14))
    gs_left = gridspec.GridSpec(1, 1)
    gs_left.update(left=0.02, right=1.)
    # plotting window for n(kx,ky)
    axM = fig.add_subplot(gs_left[0])
    axM.set_xlabel(r"M")
    axM.set_ylabel(r"T")
    axM.set_xlim([M_min,M_max*dM])
    axM.set_ylim([T_min,T_max])
    axM.set_xscale('log')
    axM.set_yscale('log')
    
    gs_right = gridspec.GridSpec(1, 1)
    gs_right.update(left=0.70, right=0.80)
    axCB = fig.add_subplot(gs_right[0])
    axCB.set_ylabel(r'$n_C/n$')
    axCB.set_xlim([0,3])
    axCB.set_ylim([0,1])
    axCB.set_xticks([0,1,2,3])
    axCB.set_yticks([0,0.5,1])
    
    
    #------------------------plot n(M,T_e)-------------------------------------
    # norm for colorbar
    norm = cm.colors.Normalize(vmax=n_max, vmin=n_min)
    # bitmap for plotting 
    bitmap = 255 * np.ones((N_T, N_M, 3), dtype = np.uint8)
    # inverse RGB-colors for the condensate states
    invColor_gs = [255, 255, 0]
    invColor_1az = [0, 255, 255]
    invColor_2az = [255, 0, 255]
    # array with all colors --> eventually one has to add more colors
    N_C = 3 # number of colors
    colors = [invColor_gs, invColor_1az, invColor_2az]
    
    # calculate bitmap
    # iterate over all system sizes and fill in all columns
    for i in range(N_M):
        # iterate over all 3 components RGB
        for j in range(3): 
            # N_T-dimensional vector + N_T-dimensional vector!
            bitmap[:,i,j] += -1*colors[ind_cond[i]][j] * cond_frac[:,i]

    extent = [M_min, M_max*dM, T_min, T_max]
    graph_M = axM.imshow(bitmap, interpolation = 'None', extent = extent)
    
    # create the colorbar
    N_n = 100           # number of colorbar-occupationnumber-fractions
    # sample-points for the colorbar
    cbx, cby = np.meshgrid(range(N_C),np.linspace(0,1,N_n))
    cb_bitmap = 255 * np.ones((N_n, N_C, 3), dtype = np.uint8)
    # calculate colorbar bitmap
    # iterate over all system sizes and fill in all columns
    for i in range(N_C):
        # iterate over all 3 components RGB
        for j in range(3): 
            # N_T-dimensional vector + N_T-dimensional vector!
            cb_bitmap[:,i,j] += -1*colors[i][j] * cby[:,i]
    extent_cb = [0, 3, 0, 1]
    graph_cb = axCB.imshow(cb_bitmap, interpolation = 'None', 
                           extent = extent_cb)
    axM.set_aspect(1)
    # optimize font-size   
    matplotlib.rcParams.update({'font.size': 16})
    plt.show()
    
    
if __name__ == '__main__':
   main()