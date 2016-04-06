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
            params[5], params[6], params[7]

def main():
    #-------------------------Load saved data----------------------------------
    fname_mat_M_n = 'mat_M_n.dat'     # file-name of the file for mat_M_n
    fname_Mx = 'Mx.dat'               # file-name of the file for Mx
    fname_T_e = 'T_e.dat'             # file-name of the file for T_e
    fname_params = 'params.dat'       # file-name of the file for params
    
    # physical parameters
    Jx, Jy, lx, ly, n, g_h, g_e, T_h = getParams(fname_params)   
    
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
    M = Mx * (Mx - 1)
    print ind_cond
    # graphs for imshow - rowwise mirrored. Represent the fraction of the
    # condensed particles to the total particle number
    cond_frac = mat_M_n[2::][::-1]


    #------------------------Plot Parameters-----------------------------------
    n_min = 0                         # minimal value of the occupation number
    n_max = 1                         # maximal value of the occupation number
    M_min = Mx[0]                     # minmal system size
    M_max = Mx[-1]                    # maximal system size 
    T_min = T_e[0]                    # minimal temperature
    T_max = T_e[-1]                   # maximal temperature
    
    # bitmap for plotting 
    bitmap = 255 * np.ones((N_T, N_M, 3), dtype = np.uint8)
    
    # inverse RGB-colors for the condensate states 
    # gs = ground state, es = excited state
                                    # color in the plot <-> complementary color
    compColor_gs = [255, 255, 0]    # blue              <-> yellow
    compColor_1es = [0, 255, 255]   # red               <-> cyan 
    compColor_2es = [255, 0, 255]   # green             <-> magenta
    compColor_3es = [255, 0, 0]     # cyan              <-> red
    compColor_4es = [0, 255, 0]     # magenta           <-> green
    compColor_5es = [0, 0, 255]     # yellow            <-> blue
    # array with all colors --> eventually one has to add more colors
    colors = [compColor_gs, compColor_1es, compColor_2es,
              compColor_3es, compColor_4es, compColor_5es]
    N_C = len(colors) # number of colors
    # extents for imshows
    extent_M = [M_min, M_max+1, T_min, T_max]
    extent_cb = [0, N_C, 0, 1]
    
    #------------------------set - up plotting windows-------------------------
    fig = plt.figure("Mean-field occupation", figsize=(16,14))
    gs_left = gridspec.GridSpec(1, 1)
    gs_left.update(left=0.1, right=0.68)
    
    # plotting window for n(kx,ky)
    axM = fig.add_subplot(gs_left[0])
    axM.set_xlabel(r"M")
    axM.set_ylabel(r"T")
    axM.set_xlim([M_min,M_max + 1])
    axM.set_ylim([T_min,T_max])
    axM.set_yscale('log')
    
    # plotting window for colorbar
    gs_right = gridspec.GridSpec(1, 1)
    gs_right.update(left=0.75, right=0.98)
    axCB = fig.add_subplot(gs_right[0])
    axCB.set_ylabel(r'$n_C/n$')
    axCB.set_xlim([0,N_C])
    axCB.set_ylim([n_min, n_max])
    axCB.set_xticks([])
    axCB.set_yticks([0,0.5,1])
    

    #------------------------plot n(M,T_e)-------------------------------------    
    # calculate bitmap
    # iterate over all system sizes and fill in all columns
    for i in range(N_M):
        # iterate over all 3 components RGB
        for j in range(3): 
            # N_T-dimensional vector + N_T-dimensional vector!
            bitmap[:,i,j] += -1*colors[ind_cond[i]][j] * cond_frac[:,i]
    graph_M = axM.imshow(bitmap, interpolation = 'None', extent = extent_M)
    
    # create the colorbar
    N_n = 100           # number of colorbar-occupationnumber-fractions
    # sample-points for the colorbar
    cbx, cby = np.meshgrid(range(N_C),np.linspace(n_min, n_max, N_n))
    cb_bitmap = 255 * np.ones((N_n, N_C, 3), dtype = np.uint8)
    # calculate colorbar bitmap
    # iterate over all system sizes and fill in all columns
    for i in range(N_C):
        # iterate over all 3 components RGB
        for j in range(3): 
            # N_T-dimensional vector + N_T-dimensional vector!
            cb_bitmap[:,i,j] += -1*colors[i][j] * cby[:,i]
    graph_cb = axCB.imshow(cb_bitmap[::-1], interpolation = 'None', 
                           extent = extent_cb)

    # optimize font-size   
    matplotlib.rcParams.update({'font.size': 16})
    plt.show()
    
if __name__ == '__main__':
   main()