import numpy as np
from scipy import optimize as opt


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

def n_bose(E, beta, z):
    return z / (np.exp(beta * E) - z)

def findz(N, E, beta):
    z_0 = np.exp(beta * E[0])
    func = lambda z: (np.sum(n_bose(E[1:], beta, z)) + z/(z_0-z) - N) #** 2. + (z > z_0) * 1000. * (z-z_0)
    z = opt.fsolve(func, z_0* N / (1 + N))
    if z >= z_0:
        func = lambda z: (np.sum(n_bose(E[1:], beta, z_0)) + z/(z_0-z) - N) #** 2. + (z > z_0) * 1000. * (z-z_0)
        z = opt.fsolve(func, z_0*N / (1 + N))
    return z

def get_eq_mfo(Jx, Jy, Mx, My, lx, ly, n, g_e, T_h, N_T, T_min, T_max):
    """ Calculates the mean occupation numbers for an ideal bose gas
        in thermal equibrilium in a one-bath-system in dependency on the
        environment-temperature T.
        Returns a matrix with ni(T)."""
    M = Mx * My                         # new 2D-system size
    N = n*M                             # number of particles
    # temperature sample points 
    T = np.logspace(np.log10(T_min), np.log10(T_max), N_T) 
    kx = get_k(Mx)                      # vector of all quasimomenta in x-dir
    ky = get_k(My)                      # vector of all quasimomenta in y-dir
    kx_plot = kx[2]                     # initial quasimomentum in x-dir
    ky_plot = ky[0]                     # initial quasimomentum in y-dir
    k = get_vec_k(kx, ky, Mx, My)       # vector of tuples of (kx,ky)
    E = get_E(k, Jx, Jy, M)
    
    mat_n = np.zeros((M, N_T))
    for i in range(N_T):
        z = findz(N, E, 1./T[i])
        n_res = n_bose(E, 1./T[i], z)
        mat_n[:,i] = n_res
    return mat_n


