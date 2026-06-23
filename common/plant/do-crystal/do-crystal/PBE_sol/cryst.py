'''
Define functions used for calculation of crystallization
'''
import casadi as ca


# define some agglomeration kernels
def constant_kernel(L1: float, L2: float, beta: float) -> float:
    return beta


def sum_kernel(L1: float, L2: float, beta: float) -> float:
    return (L1 + L2) * beta

def custom_kernel(L1: float, L2: float, beta: float) -> float:
    return 1/(L1 + L2)**2 * beta


# define functions for crystallization
def solubility(T: float) -> float:
    '''
    Calculate solubility for given temperature.
    '''
    return 0.11238 * ca.exp(9.0849e-3 * (T - 273.15))  # Parameters from Wohlgemuth 2012 for L-Alanine/ water system


def G(rel_S: float) -> float:
    '''
    Calculate growth rate for given relative supersaturation.
    '''
    return 5.857e-5 * rel_S ** 2 * ca.tanh(0.913 / rel_S) # Parameters from Hohmann et al. 2018


def nucl(rel_S: float) -> float:
    '''
    Calculate nucleation rate for given relative supersaturation.
    '''
    return 1e-1 * ca.exp(-5e-2 / (ca.log(rel_S + 1) ** 2))  # dummy value


def beta(rel_S: float) -> float:
    '''
    Calculate agglomeration rate for given relative supersaturation.
    '''
    return 1e-10 * rel_S ** 2 * ca.exp(rel_S)  # used for first case study performed for tubular

def slug_length(roh_l, v_0, eta_l, d_sfc, sigma_l, eps_0):
    # fitted parameters
    c1 = 1.969
    c2 = -1.102
    c3 = -0.035
    c4 = -0.176
    c5 = -0.0605

    Re = roh_l * v_0 * d_sfc / (eta_l)
    Ca = (eta_l * v_0) / sigma_l

    L_s = d_sfc * (c1 * (1 - eps_0) ** c2 * eps_0 ** c3 * Re ** c4 * Ca ** c5)
    L_g = L_s * (1 - eps_0) / eps_0
    L_UC = L_s + L_g
    return L_s