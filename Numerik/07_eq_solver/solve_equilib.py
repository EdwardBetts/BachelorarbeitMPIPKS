import numpy as np
from scipy import optimize as opt
from matplotlib import pyplot as plt

def Ener(k, J):
    return 2 * J * (1 - np.cos(k))

def n_bose(k, E, beta, z):
    return z / (np.exp(beta * E(k)) - z)

def findz(k, E, beta, n):
    M = len(k)
    N = n * M
    z_0 = np.exp(beta * E(k[0]))
    func = lambda z: (np.sum(n_bose(k[1:], E, beta, z)) + z/(z_0-z) - N) ** 2. + (z > z_0) * 1000. * (z-z_0)
    z = opt.fsolve(func, N / (1 + N))
    if z >= z_0:
        func = lambda z: (np.sum(n_bose(k[1:], E, beta, z_0)) + z/(z_0-z) - N) ** 2. + (z > z_0) * 1000. * (z-z_0)
        z = opt.fsolve(func, N / (1 + N))
    return z

def main():
    M = 100
    J = 1.
    n = 3.
    T = 1.
    
    k = (np.arange(M) + 1.) / (M + 1.) * np.pi
    E = lambda k: Ener(k, J)
    z = findz(k, E, 1./T, n)
    n_res = n_bose((np.arange(M) + 1.) * np.pi / (M + 1.), E, 1./T, z)
    
    plt.semilogy(k, n_res)
    plt.show()

if __name__ == "__main__":
    main()