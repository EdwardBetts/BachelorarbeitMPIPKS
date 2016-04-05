# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 14:52:30 2016

@author: toni
"""
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.cm as cm

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
    # graphs for imshow - rowwise mirrored 
    graphs = mat_M_n[2::][::-1]
    slices = get_conn_parts(ind_cond) 
    print Mx[slices[0]]
    
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
    
    # plotting window for n(kx,ky)
    axM = fig.add_subplot(111)
    axM.set_xlabel(r"M")
    axM.set_ylabel(r"T")
    axM.set_xlim([M_min,M_max*dM])
    axM.set_ylim([T_min,T_max])
    axM.set_xscale('log')
    axM.set_yscale('log')
    
    #------------------------plot n(M,T_e)-------------------------------------
    # norm for colorbar
    norm = cm.colors.Normalize(vmax=n_max, vmin=n_min)
    
    # plotting-graph: extent has factor dM such that the borders fit
    for i in range(len(slices)):
        index = ind_cond[i]%len(cmaps)
        cmap = plt.get_cmap(cmaps[i])
        extent = [Mx[slices[i]][0], Mx[slices[i]][-1], T_min, T_max]
        graph_M = axM.imshow(graphs[:,slices[i]], interpolation = 'None', 
                             cmap = cmap, norm = norm, extent = extent)
        cb_axE = fig.colorbar(graph_M,ticks=[0, 0.5, 1],ax=axM, format='%.1f')

    # optimize font-size   
    matplotlib.rcParams.update({'font.size': 16})
    plt.show()
    
if __name__ == '__main__':
   main()