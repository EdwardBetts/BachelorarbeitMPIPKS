# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 14:52:30 2016

@author: toni
"""
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.cm as cm

def main():
    #-------------------------Load saved data----------------------------------
    fname_mat_M_n = 'mat_M_n.dat'     # file-name of the file for mat_M_n
    fname_Mx = 'Mx.dat'               # file-name of the file for Mx
    fname_T_e = 'T_e.dat'             # file-name of the file for T_e
    T_e = np.fromfile(fname_T_e, sep=';')
    N_T = len(T_e)
    Mx = np.fromfile(fname_Mx, dtype=int, sep=';')
    N_M = len(Mx)
    mat_M_n = np.fromfile(fname_mat_M_n, sep=';').reshape(N_T,N_M)
    print Mx
    
    #------------------------Plot Parameters-----------------------------------
    n_min = 0                         # minimal value of the occupation number
    n_max = 1                         # maximal value of the occupation number
    M_min = Mx[0]                     # minmal system size
    M_max = Mx[-1]                    # maximal system size
    dM = np.float(Mx[1])/Mx[0]        # 'distance' between nighboured sizes 
    T_min = T_e[0]                    # minimal temperature
    T_max = T_e[-1]                   # maximal temperature
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
    graph_M = axM.imshow(mat_M_n[::-1],interpolation = 'None', cmap = cm.binary, 
               norm =norm, extent=[M_min, M_max*dM, T_min, T_max])
    cb_axE = fig.colorbar(graph_M,ticks=[0, 0.5, 1],ax=axM, format='%.1f')

    # optimize font-size   
    matplotlib.rcParams.update({'font.size': 16})
    plt.show()
    
if __name__ == '__main__':
   main()