import casadi as ca
import numpy as np

def first_order(u_bar : ca.SX, u_in : ca.SX) -> ca.SX:
    '''
    Calculate flux on cell faces using first-order approximation
    '''

    flux = ca.vertcat(u_in, u_bar)

    return flux

def WENO5(u_bar: ca.SX, u_in: ca.SX) -> ca.SX:
    # ic('WENO5')
    '''
    Function to calculate flux on cell faces using 5th order WENO.

    Parameters:
    u_bar : ca.SX
        Array of values at cell centers.
    u_in : ca.SX
        input at left boundary.
    '''
    # linear weights gamma
    gamma_1, gamma_2, gamma_3 = 1 / 10, 3 / 5, 3 / 10

    n = u_bar.shape[0]

    flux = ca.SX.zeros(n + 1)

    flux[0] = u_in
    flux[1] = u_bar[0]

    # for u_in
    s1 = 1 / 3 * u_in - 7 / 6 * u_bar[0] + 11 / 6 * u_bar[1]
    s2 = -1 / 6 * u_bar[0] + 5 / 6 * u_bar[1] + 1 / 3 * u_bar[2]
    s3 = 1 / 3 * u_bar[1] + 5 / 6 * u_bar[2] - 1 / 6 * u_bar[3]

    # smoothness indicators beta
    beta_1 = 13 / 12 * (u_in - 2 * u_bar[0] + u_bar[1]) ** 2 + 1 / 4 * (u_in - 4 * u_bar[0] + 3 * u_bar[1]) ** 2
    beta_2 = 13 / 12 * (u_bar[0] - 2 * u_bar[1] + u_bar[2]) ** 2 + 1 / 4 * (u_bar[0] - u_bar[2]) ** 2
    beta_3 = 13 / 12 * (u_bar[1] - 2 * u_bar[2] + u_bar[3]) ** 2 + 1 / 4 * (3 * u_bar[1] - 4 * u_bar[2] + u_bar[3]) ** 2

    # smoothness indicators alpha
    alpha_1 = gamma_1 / (beta_1 + 1e-6) ** 2
    alpha_2 = gamma_2 / (beta_2 + 1e-6) ** 2
    alpha_3 = gamma_3 / (beta_3 + 1e-6) ** 2

    # nonlinear weights w
    w_1 = alpha_1 / (alpha_1 + alpha_2 + alpha_3)
    w_2 = alpha_2 / (alpha_1 + alpha_2 + alpha_3)
    w_3 = alpha_3 / (alpha_1 + alpha_2 + alpha_3)

    # WENO reconstruction
    flux[2] = w_1 * s1 + w_2 * s2 + w_3 * s3

    for i in range(2, n - 2):
        # stencil 1
        s1 = 1 / 3 * u_bar[i - 2] - 7 / 6 * u_bar[i - 1] + 11 / 6 * u_bar[i]
        # stencil 2
        s2 = -1 / 6 * u_bar[i - 1] + 5 / 6 * u_bar[i] + 1 / 3 * u_bar[i + 1]
        # stencil 3
        s3 = 1 / 3 * u_bar[i] + 5 / 6 * u_bar[i + 1] - 1 / 6 * u_bar[i + 2]

        # smoothness indicators beta
        beta_1 = 13 / 12 * (u_bar[i - 2] - 2 * u_bar[i - 1] + u_bar[i]) ** 2 + 1 / 4 * (
                    u_bar[i - 2] - 4 * u_bar[i - 1] + 3 * u_bar[i]) ** 2
        beta_2 = 13 / 12 * (u_bar[i - 1] - 2 * u_bar[i] + u_bar[i + 1]) ** 2 + 1 / 4 * (
                    u_bar[i - 1] - u_bar[i + 1]) ** 2
        beta_3 = 13 / 12 * (u_bar[i] - 2 * u_bar[i + 1] + u_bar[i + 2]) ** 2 + 1 / 4 * (
                    3 * u_bar[i] - 4 * u_bar[i + 1] + u_bar[i + 2]) ** 2

        # smoothness indicators alpha
        alpha_1 = gamma_1 / (beta_1 + 1e-6) ** 2
        alpha_2 = gamma_2 / (beta_2 + 1e-6) ** 2
        alpha_3 = gamma_3 / (beta_3 + 1e-6) ** 2

        # nonlinear weights w
        w_1 = alpha_1 / (alpha_1 + alpha_2 + alpha_3)
        w_2 = alpha_2 / (alpha_1 + alpha_2 + alpha_3)
        w_3 = alpha_3 / (alpha_1 + alpha_2 + alpha_3)

        # WENO reconstruction
        flux[i + 1] = w_1 * s1 + w_2 * s2 + w_3 * s3

    # last elements
    flux[-2] = u_bar[-2]
    flux[-1] = u_bar[-1]

    return flux

def WENO5_np(u_bar: np.ndarray, u_in: float) -> np.ndarray:
    """
    Function to calculate flux on cell faces using 5th order WENO.

    Parameters:
    u_bar : np.ndarray
        Array of values at cell centers.
    u_in : float
        input at left boundary.

    Returns:
    np.ndarray: Computed flux at cell faces
    """
    # linear weights gamma
    gamma_1, gamma_2, gamma_3 = 1/10, 3/5, 3/10

    n = len(u_bar)
    flux = np.zeros(n + 1)

    # Set boundary conditions
    flux[0] = u_in
    flux[1] = u_bar[0]

    # Calculate first interior point
    s1 = 1/3 * u_in - 7/6 * u_bar[0] + 11/6 * u_bar[1]
    s2 = -1/6 * u_bar[0] + 5/6 * u_bar[1] + 1/3 * u_bar[2]
    s3 = 1/3 * u_bar[1] + 5/6 * u_bar[2] - 1/6 * u_bar[3]

    # smoothness indicators beta
    beta_1 = 13/12 * (u_in - 2*u_bar[0] + u_bar[1])**2 + \
             1/4 * (u_in - 4*u_bar[0] + 3*u_bar[1])**2
    beta_2 = 13/12 * (u_bar[0] - 2*u_bar[1] + u_bar[2])**2 + \
             1/4 * (u_bar[0] - u_bar[2])**2
    beta_3 = 13/12 * (u_bar[1] - 2*u_bar[2] + u_bar[3])**2 + \
             1/4 * (3*u_bar[1] - 4*u_bar[2] + u_bar[3])**2

    # smoothness indicators alpha
    eps = 1e-6
    alpha_1 = gamma_1 / (beta_1 + eps)**2
    alpha_2 = gamma_2 / (beta_2 + eps)**2
    alpha_3 = gamma_3 / (beta_3 + eps)**2

    # nonlinear weights w
    alpha_sum = alpha_1 + alpha_2 + alpha_3
    w_1 = alpha_1 / alpha_sum
    w_2 = alpha_2 / alpha_sum
    w_3 = alpha_3 / alpha_sum

    # WENO reconstruction
    flux[2] = w_1 * s1 + w_2 * s2 + w_3 * s3

    # Interior points
    for i in range(2, n-2):
        # stencils
        s1 = 1/3 * u_bar[i-2] - 7/6 * u_bar[i-1] + 11/6 * u_bar[i]
        s2 = -1/6 * u_bar[i-1] + 5/6 * u_bar[i] + 1/3 * u_bar[i+1]
        s3 = 1/3 * u_bar[i] + 5/6 * u_bar[i+1] - 1/6 * u_bar[i+2]

        # smoothness indicators beta
        beta_1 = 13/12 * (u_bar[i-2] - 2*u_bar[i-1] + u_bar[i])**2 + \
                 1/4 * (u_bar[i-2] - 4*u_bar[i-1] + 3*u_bar[i])**2
        beta_2 = 13/12 * (u_bar[i-1] - 2*u_bar[i] + u_bar[i+1])**2 + \
                 1/4 * (u_bar[i-1] - u_bar[i+1])**2
        beta_3 = 13/12 * (u_bar[i] - 2*u_bar[i+1] + u_bar[i+2])**2 + \
                 1/4 * (3*u_bar[i] - 4*u_bar[i+1] + u_bar[i+2])**2

        # smoothness indicators alpha
        alpha_1 = gamma_1 / (beta_1 + eps)**2
        alpha_2 = gamma_2 / (beta_2 + eps)**2
        alpha_3 = gamma_3 / (beta_3 + eps)**2

        # nonlinear weights w
        alpha_sum = alpha_1 + alpha_2 + alpha_3
        w_1 = alpha_1 / alpha_sum
        w_2 = alpha_2 / alpha_sum
        w_3 = alpha_3 / alpha_sum

        # WENO reconstruction
        flux[i+1] = w_1 * s1 + w_2 * s2 + w_3 * s3

    # Set last elements
    flux[-2] = u_bar[-2]
    flux[-1] = u_bar[-1]

    return flux

def diffusion(u_bar: ca.SX) -> ca.SX:
    # generate doc string using github copilot
    n = u_bar.shape[0]

    diffusion = ca.SX.zeros(n)

    # no diffusion for first and last element

    # second element
    diffusion[1] = u_bar[2] - 2 * u_bar[1] + u_bar[0]
    for i in range(2, n - 2):
        diffusion[i] = (-u_bar[i + 2] + 16 * u_bar[i + 1] - 30 * u_bar[i] + 16 * u_bar[i - 1] - u_bar[i - 2]) / 12
    diffusion[-2] = u_bar[-1] - 2 * u_bar[-2] + u_bar[-3]
    return diffusion
