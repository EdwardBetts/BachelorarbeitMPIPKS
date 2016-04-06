import numpy as np
from scipy import optimize as opt
from matplotlib import pyplot as plt
import matplotlib


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
    func = lambda z: (np.sum(n_bose(E[1:], beta, z)) + z/(z_0-z) - N) ** 2. + (z > z_0) * 1000. * (z-z_0)
    z = opt.fsolve(func, z_0*N / (1 + N))
    if z >= z_0:
        func = lambda z: (np.sum(n_bose(E[1:], beta, z_0)) + z/(z_0-z) - N) ** 2. + (z > z_0) * 1000. * (z-z_0)
        z = opt.fsolve(func, z_0*N / (1 + N))
    return z
    
def plot_axT(T, M, mat_n):
    """ Plots the occupation numbers for different environment temperatures."""
    for i in range(M):
        axT.plot(T,np.abs(mat_n[i,:]), c = 'b')  
        
def plot_init_axK(T, E, k, N_T, mat_n):
    """Plot """
    global graph_K
    global vline
    #dist_T = np.abs(T - T[N_T/2])
    #ind_plot = np.arange(N_T)[dist_T == np.min(dist_T)][0]
    T_plot = T[N_T/2]
    vline = axT.axvline(T_plot, color='r')
    graph_K = axK.plot(k, mat_n[:,N_T/2], color='r')


def onMouseClick(event):
    mode = plt.get_current_fig_manager().toolbar.mode
    if  event.button == 1 and event.inaxes == axT and mode == '':
        global graph_K
        global vline
        # remove lines drawn before        
        g = graph_K.pop(0)
        g.remove()
        del g
        vline.remove()
        del vline    
        
        # find the clicked position and draw a vertical line
        T_click = event.xdata
        dist_T = np.abs(T - T_click)
        ind_click = np.arange(N_T)[dist_T == np.min(dist_T)][0]
        T_plot = T[ind_click]
        vline = axT.axvline(T_plot, color='r')
        
        # plot n(k)
        #print mat_n[:,ind_click]
        graph_K = axK.plot(k, mat_n[:,ind_click], color='r')
        # refreshing
        fig.canvas.draw()


#---------------------------physical parameters--------------------------------
Jx = 1.                             # dispersion-constant in x-direction
Jy = 1.                             # dispersion-constant in y-direction   
Mx = 20                             # system size in x-direction
My = 30                             # system size in y-direction
lx = 4.                             # heated site (x-component)
ly = 3.                             # heated site (y-component)
n = 3                               # particle density
g_h = 0.                            # coupling strength needle<->system
g_e = 1.                            # coupling strength environment<->sys
T_h = 60*Jx                         # temperature of the needle
M = Mx * My                         # new 2D-system size
N = n*M                             # number of particles

#----------------------------program parameters--------------------------------
N_T = 100                           # number of temp. data-points
T = np.logspace(-2,2,N_T)

vline = None
graph_K = None

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
print mat_n

fig = plt.figure("Mean-field occupation", figsize=(16,9))
axT = fig.add_subplot(121)
axT.set_xlabel(r'$T/J$')
axT.set_ylabel(r'$\bar{n}_i$')
axT.set_xlim([np.min(T), np.max(T)])
axT.set_ylim([8*10e-5, 3*N])
axT.set_xscale('log')
axT.set_yscale('log')
    
axK = fig.add_subplot(122)
axK.set_xlabel(r'$k/a$')
axK.set_ylabel(r'$\bar{n}_i$')
axK.set_xlim([0, np.pi])
axK.set_ylim([8*10e-4, 3*N])
axK.set_yscale('log')

# plot initial lines    
plot_axT(T, M, mat_n)
plot_init_axK(T, E, k, N_T, mat_n)
# refreshing
fig.canvas.draw()

# connect plotting window with the onClick method
cid = fig.canvas.mpl_connect('button_press_event', onMouseClick)
    
matplotlib.rcParams.update({'font.size': 18})
plt.show()

