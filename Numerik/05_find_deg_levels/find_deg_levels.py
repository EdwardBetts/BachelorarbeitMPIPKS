# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 13:30:45 2016

@author: toni
"""
import numpy as np

def get_k(M):
    """ Return all M possible quasimomenta of the tight binding chain
    in the first Brillouin zone."""
    k = np.arange(1,M+1)*np.pi/(M+1)    # vector of all possible quasimomenta
    return k

def get_degen_levels(kx):
    c = np.cos(kx)
    mat_k1, mat_k2 = np.meshgrid(kx,kx)
    mat_c1, mat_c2 = np.meshgrid(c,c)
    c_diff = np.abs(mat_c1 - mat_c2)
    if np.any(np.abs(c_diff-np.sqrt(2)) <= 1e-14) or np.any(np.abs(c_diff-np.sqrt(2)/2.) <= 1e-14):
        print len(kx)
    
    
M = np.arange(1000)
for m in M:
    get_degen_levels(get_k(m))

