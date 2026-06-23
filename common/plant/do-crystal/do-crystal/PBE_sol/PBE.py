import numpy as np
import casadi as ca
import collections.abc
import scipy
import matplotlib.pyplot as plt
import time
import PBE_sol.functions as functions
import PBE_sol.cryst as cryst



class PBE:
    '''
    Class for solution of the population balance equation (PBE) using different methods.

    Solution methods:
    - SMOM: Standard method of moments
    - QMOM: Quadrature method of moments
    - DPBE: Discrete PBE
    - OCFE: Orthogonal collocation on finite elements

    Methods:
    - setup: Set up the PBE method with given parameters specific to the method.
    - state_shape: Return the shape of the differential state vector for do-mpc for the given PBE method.
    - alg_shape: Return the shape of the algebraic state vector for do-mpc for the given PBE method.
    - rhs: Compute the right-hand side of the PBE for the given PBE method.
    - alg: Compute the algebraic equations for the given PBE method.
    - calc_moments: Calculate moments from the PSD for the given PBE method.
    '''

    def __init__(self, PBE_method: str, coordinate: str = 'L'):
        '''
        Initialize the PBE class with the given method and inner coordinate.

        Parameters:
        - method: PBE method to use. Options are 'SMOM', 'QMOM', 'DPBE', and 'OCFE'
        - coordinate: Inner coordinate to use. Options are 'L' for length and 'V' for volume
        '''
        self.method = PBE_method
        self.coordinate = coordinate
        self.setup_done = False

        if self.method == 'SMOM':
            print('Provide number of moments in setup.')
        elif self.method == 'QMOM':
            pass
        elif self.method == 'DPBE':
            print('Provide scheme, spacing, number of bins, q (geometric spacing factor), and L_0 in setup.')
        elif self.method == 'OCFE':
            print(
                'Provide number of finite elements, number of collocation points per element, n_linear, step_linear, D_a, and domain in setup.')

    def __str__(self) -> str:
        return f'Class for solution of PBE using {self.method}.'

    def setup(self, *args, **kwargs) -> None:
        '''
        Set up the PBE method with given parameters specific to the method.

        Parameters necessary for each method (kwargs):
            - SMOM:
                - n_moments: Number of moments
            - QMOM:
                - kernel: Kernel function
            - DPBE:
                necessary:
                - scheme: 'first', 'second', 'k-1', 'k1/3', 'limited'
                - spacing: 'geometric', 'uniform'
                - no_class: Number of classes
                - q: Geometric spacing factor (necessary to provide but only used for geometric spacing)
                - L_0: Lower bound of domain for geometric spacing (necessary to provide but only used for geometric spacing)
                - domain: Domain of distribution (only used for uniform spacing)
                optional:
                - G_fun_0: Growth function with unit rate
            - OCFE:
                necessary:
                - n_elements: Number of finite elements
                - n_col: Number of collocation points per element
                - n_linear: Number of linearly spaced elements (necessary to provide but only used for geometric spacing)
                - step_linear: Step size for linearly spaced elements (necessary to provide but only used for geometric spacing)
                - geometric_factor: Geometric factor for geometric spacing (necessary to provide but only used for geometric spacing)
                - D_a: Artificial diffusion coefficient for OCFE calculation
                - domain: Domain of distribution
                - boundary_cond: Boundary condition for OCFE
                optional:
                - G_fun_0: Growth function with unit rate
                - G_fun_0_deriv: Derivative of growth function with unit rate (necessary if G_fun_0 is provided)
        '''

        if self.method == 'SMOM':
            # check if kwargs contain necessary parameters
            # for SMOM necessary: n_moments
            if 'n_moments' not in kwargs:
                raise ValueError('Please provide number of moments.')

            self.n_moments = kwargs['n_moments']
            self.set_alg = False
            # if self.beta != 0:
            #     print(f'SMOM cannot consider agglomeration. Setting beta to 0.')
        elif self.method == 'QMOM':
            self.set_alg = False

            # check if kwargs contain necessary parameters
            # for QMOM necessary: kernel
            if 'kernel' not in kwargs:
                raise ValueError('Please provide kernel function.')

            if kwargs['kernel'] == "constant_kernel":
                self.kernel = cryst.constant_kernel
            else:
                print('Kernel function not recognized. Using constant kernel.')
                self.kernel = cryst.constant_kernel

            # generate casadi function for Jacobian of PD algorithm
            self.n_moments = 6  # only implemented for 6 moments
            m = ca.SX.sym('m', self.n_moments)

            P = ca.SX.zeros(self.n_moments + 1, 1 + self.n_moments)
            P[0, 0] = 1
            for row in range(self.n_moments):
                P[row, 1] = (-1) ** row * m[row]
            for col in range(2, self.n_moments + 1):
                for row in range(m.shape[0]):
                    P[row, col] = P[0, col - 1] * P[row + 1, col - 2] - P[0, col - 2] * P[row + 1, col - 1]

            alpha = ca.SX.zeros(self.n_moments)
            for i in range(1, self.n_moments):
                alpha[i] = P[0, i + 1] / (P[0, i] * P[0, i - 1])

            a = ca.SX.zeros(int(self.n_moments / 2))
            for i in range(a.shape[0]):
                a[i] = alpha[2 * i] + alpha[2 * i + 1]

            b = ca.SX.zeros(int(self.n_moments / 2))
            for i in range(a.shape[0] - 1):
                b[i] = ca.sqrt(alpha[2 * i + 1] * alpha[2 * i + 2])

            J = ca.SX.zeros(int(self.n_moments / 2), int(self.n_moments / 2))
            for row in range(int(self.n_moments / 2)):
                J[row, row] = a[row]

            J[0, 1] = b[0]
            J[-1, -2] = b[-2]
            for row in range(1, int(self.n_moments / 2) - 1):
                for col in range(int(self.n_moments / 2)):
                    J[row, row - 1] = b[row - 1]
                    J[row, row + 1] = b[row]

            self.J_fun = ca.Function('J', [m], [J])

            # define algorithm to compute symbolic eigenvectors of J as casadi function given J and eigval
            eigval = ca.SX.sym('eigval', 3)

            d = self.J_fun(m)[1, 0]
            e = self.J_fun(m)[1, 1]
            f = self.J_fun(m)[1, 2]
            g = self.J_fun(m)[2, 0]
            h = self.J_fun(m)[2, 1]
            i = self.J_fun(m)[2, 2]

            # d, e-lamb,f   g,h, i-lamb
            vector0 = ca.SX.zeros(3)
            vector1 = ca.SX.zeros(3)
            vector2 = ca.SX.zeros(3)

            vector0[0] = (e - eigval[0]) * (i - eigval[0]) - f * h
            vector0[1] = f * g - d * (i - eigval[0])
            vector0[2] = d * h - (e - eigval[0]) * g

            vector1[0] = (e - eigval[1]) * (i - eigval[1]) - f * h
            vector1[1] = f * g - d * (i - eigval[1])
            vector1[2] = d * h - (e - eigval[1]) * g

            vector2[0] = (e - eigval[2]) * (i - eigval[2]) - f * h
            vector2[1] = f * g - d * (i - eigval[2])
            vector2[2] = d * h - (e - eigval[2]) * g

            out = ca.vertcat(vector0[0] / ca.norm_2(vector0), vector1[0] / ca.norm_2(vector1),
                             vector2[0] / ca.norm_2(vector2))  # only need first component of eigenvectors

            self.eigvec_fun = ca.Function('eigvec_fun', [m, eigval], [out])

            # define sums needed for calculation of agglomeration and breakage as casadi functions
            L = ca.SX.sym('L', 3)
            w = ca.SX.sym('w', 3)

            k = ca.SX.sym('k')
            beta = ca.SX.sym('beta')

            agg = 0
            if self.coordinate == 'L':
                print('L coordinate')
                for i in range(3):
                    for j in range(3):
                        agg += 0.5 * w[i] * w[j] * (L[i] ** 3 + L[j] ** 3) ** (k / 3) * self.kernel(L[i], L[j], beta)
                        agg -= L[i] ** k * w[i] * w[j] * self.kernel(L[i], L[j], beta)

                self.agg_fun = ca.Function('agg_fun', [L, w, k, beta], [agg])
            elif self.coordinate == 'V':
                print('V coordinate')
                for i in range(3):
                    for j in range(3):
                        agg += 0.5 * w[i] * w[j] * (L[i] + L[j]) ** k * self.kernel(L[i], L[j], beta)
                        agg -= L[i] ** k * w[i] * w[j] * self.kernel(L[i], L[j], beta)

                self.agg_fun = ca.Function('agg_fun', [L, w, k, beta], [agg])
            else:
                raise ValueError('Invalid coordinate. Please choose from L or V.')


        elif self.method == 'DPBE':

            # check if kwargs contain necessary parameters
            # for DPBE necessary: scheme, spacing, no_class, q, L_0, domain
            if 'scheme' not in kwargs:
                raise ValueError('Please provide scheme.')
            if 'spacing' not in kwargs:
                raise ValueError('Please provide spacing.')
            if 'no_class' not in kwargs:
                raise ValueError('Please provide number of classes.')
            if 'q' not in kwargs:
                raise ValueError('Please provide q.')  # this can be technically be omitted if spacing is uniform
            if 'L_0' not in kwargs:
                raise ValueError('Please provide L_0.')
            if 'domain' not in kwargs:
                raise ValueError('Please provide domain.')

            self.scheme = kwargs['scheme']
            self.spacing = kwargs['spacing']
            self.no_class = kwargs['no_class']
            self.q = kwargs['q']
            self.L_0 = kwargs['L_0']
            self.domain = kwargs['domain']
            self.set_alg = False

            # check if growth function is provided otherwise use constant growth
            # G_fun_0 is growth function with unit rate
            if 'G_fun_0' in kwargs:
                self.G_fun_0 = kwargs['G_fun_0']
                print('Growth function provided.')
            else:
                self.G_fun_0 = lambda _: 1
                print('No growth function provided. Using constant growth.')

            # initialize discrete grid
            if self.spacing == 'geometric':
                del_ = 2 ** (1 / (3 * self.q))  # geometric ratio
                self.L_i = np.zeros(self.no_class)
                self.L_i[0] = self.L_0
                for i in range(1, self.no_class):
                    self.L_i[i] = self.L_i[i - 1] * del_
                if self.coordinate == 'L':
                    q = self.q
                    self.q = q
                elif self.coordinate == 'V':
                    q = self.q
                    self.q = 3 * q
            if self.spacing == 'uniform':
                self.L_i = np.linspace(*self.domain, self.no_class)

            # calculate length of each class
            self.del_L_i = np.concatenate((np.diff(self.L_i), np.diff(self.L_i)[0].reshape(-1)))

            # update domain
            self.domain = [self.L_i[0], self.L_i[-1]]

            # shift L_i to center of class
            self.L_i_bound = np.concatenate((self.L_i, [self.L_i[-1] + self.del_L_i[-1]]))
            self.L_i = self.L_i + self.del_L_i / 2

            print(f'Grid consisting of {self.no_class} classes with {self.spacing} spacing.')
            print(f'Domain of grid: [{self.domain[0]},{self.domain[1]}]')

            if self.coordinate == 'L':
                self.map = np.zeros((self.no_class, self.no_class), dtype=int)
                for i in range(self.no_class):
                    for j in range(i):
                        d_check = (self.L_i[i] ** 3 - self.L_i[j] ** 3) ** (1 / 3)
                        self.map[i, j] = np.argmin(np.fabs(self.L_i - d_check))

        elif self.method == 'OCFE':

            # check if kwargs contain necessary parameters
            # for OCFE necessary: n_elements, n_col, n_linear, step_linear, D_a, domain, boundary_cond
            if 'n_elements' not in kwargs:
                raise ValueError('Please provide number of finite elements.')
            if 'n_col' not in kwargs:
                raise ValueError('Please provide number of collocation points per element.')
            # if 'n_linear' not in kwargs:
            #     raise ValueError('Please provide number of linearly spaced elements.')
            # if 'step_linear' not in kwargs:
            #     raise ValueError('Please provide step size for linearly spaced elements.')
            # if 'geometric_factor' not in kwargs:
            #     raise ValueError('Please provide geometric factor.')
            if 'D_a' not in kwargs:
                raise ValueError('Please provide artificial diffusion coefficient.')
            if 'domain' not in kwargs:
                raise ValueError('Please provide domain.')
            if 'boundary_condition' not in kwargs:
                raise ValueError('Please provide boundary condition.')

            self.n_elements = kwargs['n_elements']
            self.n_col = kwargs['n_col']

            # do not use geometric spacing --> n_linear = n_elements
            self.n_linear = self.n_elements
            self.step_linear = (kwargs['domain'][1] - kwargs['domain'][0]) / self.n_elements
            self.geometric_factor = 0

            self.D_a = kwargs['D_a']
            self.domain = kwargs['domain']
            self.boundary_condition = kwargs['boundary_condition']

            self.set_alg = True

            # check if growth function is provided otherwise use constant growth
            # G_fun_0 is growth function with unit rate
            # for OCFE derivative of growth function is needed
            if 'G_fun_0' in kwargs:
                self.G_fun_0 = kwargs['G_fun_0']
                self.G_fun_0_deriv = kwargs['G_fun_0_deriv']
                print('Growth function provided.')
            else:
                self.G_fun_0 = lambda _: 1
                self.G_fun_0_deriv = lambda _: 0
                print('No growth function provided. Using constant growth.')

            col = list(((scipy.special.roots_legendre(self.n_col - 2)[0] + 1) / 2))

            # weights for integration using gauss-legendre quadrature on [0,1]
            _, weights2 = scipy.special.roots_legendre(self.n_col - 2)
            self.weights = weights2 / 2

            self.z = [0] + col + [1]  # collocation points

            self.z_outer = np.zeros(self.n_elements + 1)
            self.step = np.zeros(self.n_elements)
            self.z_outer[0] = self.domain[0]

            for i in range(1, self.n_elements + 1):
                if i <= self.n_linear:
                    self.z_outer[i] = self.z_outer[i - 1] + self.step_linear
                    self.step[i - 1] = self.step_linear
                else:
                    self.z_outer[i] = self.z_outer[i - 1] * self.geometric_factor
                    self.step[i - 1] = self.z_outer[i] - self.z_outer[i - 1]

            self.n_pos_full = np.zeros((self.n_elements, self.n_col))
            for element in range(self.n_elements):
                self.n_pos_full[element, :] = np.array(
                    [self.z_outer[element] + z_i * self.step[element] for z_i in self.z])

            self.n_pos = self.n_pos_full[:, 1:self.n_col - 1]
            print(
                f'Generated grid with {self.n_linear} linearly spaced element/s and {self.n_elements - self.n_linear} geometrically spaced elements. {self.n_col * self.n_elements} collocation points in total.')

            if self.domain[1] != self.n_pos_full[-1, -1]:
                self.domain[1] = self.n_pos_full[-1, -1]
                print(f'Updated domain to {self.domain}')

            self.s = np.zeros((self.n_col, self.n_col))
            self.s_2 = np.zeros((self.n_col, self.n_col))
            self.s_3 = np.zeros((self.n_col, self.n_col))
            self.s_4 = np.zeros((self.n_col, self.n_col))
            # compute coefficients of single polynomials L_i which form n(v)=sum(n_i*L_i)
            self.coef = np.zeros((self.n_col, self.n_col))
            for j in range(self.n_col):
                for k in range(self.n_col):
                    # scipy.interpolate.lagrange yields lagrange polynomials
                    # we want to calculate the derivative at each point z_i 
                    # --> coefficients for lagrange polynomials are 1 for i=j and 0 for i!=j (np.eye[k,:])
                    # scipy.interpolate.lagrange gives old numpy polynomial object which must be converted to new numpy object
                    # --> .coef[::-1] since coefficients in old numpy implementation are reversed
                    # .deriv()(z[j]) gives derivative evaluated at z[j]
                    self.s[j, k] = np.polynomial.Polynomial(
                        scipy.interpolate.lagrange(self.z, np.eye(len(self.z))[k, :]).coef[::-1]).deriv()(self.z[j])
                    self.s_2[j, k] = np.polynomial.Polynomial(
                        scipy.interpolate.lagrange(self.z, np.eye(len(self.z))[k, :]).coef[::-1]).deriv(2)(self.z[j])
                    self.s_3[j, k] = np.polynomial.Polynomial(
                        scipy.interpolate.lagrange(self.z, np.eye(len(self.z))[k, :]).coef[::-1]).deriv(3)(self.z[j])
                    self.s_4[j, k] = np.polynomial.Polynomial(
                        scipy.interpolate.lagrange(self.z, np.eye(len(self.z))[k, :]).coef[::-1]).deriv(4)(self.z[j])

                self.coef[j, :] = np.polynomial.Polynomial(
                    scipy.interpolate.lagrange(self.z, np.eye(len(self.z))[j, :]).coef[::-1]).coef

            if self.coordinate == 'L':
                # map 1 gives element for any particle (L_i_e**3 and L_j_k**3)**(1/3)
                self.map1 = np.zeros((self.n_elements, self.n_col - 2, self.n_elements, self.n_col), dtype=int)
                for element in range(self.n_elements):
                    for col in range(self.n_col - 2):
                        for e in range(self.n_elements):
                            for k in range(self.n_col):
                                for el in range(self.n_elements):
                                    if self.n_pos[element, col] ** 3 - self.n_pos_full[e, k] ** 3 < 0:
                                        self.map1[element, col, e, k] = 0
                                    elif (self.n_pos[element, col] ** 3 - self.n_pos_full[e, k] ** 3) ** (1 / 3) >= \
                                            self.n_pos_full[el, 0] and (
                                            self.n_pos[element, col] ** 3 - self.n_pos_full[e, k] ** 3) ** (1 / 3) <= \
                                            self.n_pos_full[el, -1]:
                                        self.map1[element, col, e, k] = el
                # map 2 to get element for (L_i_e**3)/2
                # only one output per L_i_e
                self.map2 = np.zeros((self.n_elements, self.n_col - 2), dtype=int)
                for element in range(self.n_elements):
                    for col in range(self.n_col - 2):
                        for el in range(self.n_elements):
                            if self.n_pos[element, col] / (2 ** (1 / 3)) >= self.n_pos_full[el, 0] and self.n_pos[
                                element, col] / (2 ** (1 / 3)) <= self.n_pos_full[el, -1]:
                                self.map2[element, col] = el

            elif self.coordinate == 'V':
                # map 1 gives element for any difference between V_i_e and V_j_k
                self.map1 = np.zeros((self.n_elements, self.n_col - 2, self.n_elements, self.n_col), dtype=int)
                for element in range(self.n_elements):
                    for col in range(self.n_col - 2):
                        for e in range(self.n_elements):
                            for k in range(self.n_col):
                                for el in range(self.n_elements):
                                    if self.n_pos[element, col] - self.n_pos_full[e, k] < 0:
                                        self.map1[element, col, e, k] = 0
                                    elif self.n_pos[element, col] - self.n_pos_full[e, k] >= self.n_pos_full[el, 0] and \
                                            self.n_pos[element, col] - self.n_pos_full[e, k] <= self.n_pos_full[el, -1]:
                                        self.map1[element, col, e, k] = el

                # map 2 to get element for V_i_e/2
                # only one output per V_i_e
                self.map2 = np.zeros((self.n_elements, self.n_col - 2), dtype=int)
                for element in range(self.n_elements):
                    for col in range(self.n_col - 2):
                        for el in range(self.n_elements):
                            if self.n_pos[element, col] / 2 >= self.n_pos_full[el, 0] and self.n_pos[
                                element, col] / 2 <= self.n_pos_full[el, -1]:
                                self.map2[element, col] = el
        else:
            raise ValueError('Invalid PBE method. Please choose from SMOM, QMOM, DPBE, or OCFE.')

        self.setup_done = True

    def state_shape(self) -> int:
        '''
        Return the shape of the differential state vector for do-mpc for the given PBE method.
        '''
        if not (self.setup_done):
            raise ValueError('Please set up the PBE method first.')
        else:
            if self.method == 'SMOM':
                return self.n_moments,
            elif self.method == 'QMOM':
                return 6,  # only implemented for 6 moments
            elif self.method == 'DPBE':
                return self.no_class,
            elif self.method == 'OCFE':
                return self.n_elements * (self.n_col - 2),
            else:
                raise ValueError('Invalid PBE method. Please choose from SMOM, QMOM, DPBE, or OCFE.')

    def alg_shape(self) -> int:
        '''
        Return the shape of the algebraic state vector for do-mpc for the given PBE method.
        '''
        if self.method == 'SMOM':
            return 0
        elif self.method == 'QMOM':
            return 0
        elif self.method == 'DPBE':
            return 0
        elif self.method == 'OCFE':
            return (self.n_elements * 2),
        else:
            raise ValueError('Invalid PBE method. Please choose from SMOM, QMOM, DPBE, or OCFE.')

    def rhs(self, states: ca.SX, alg_states: ca.SX = 0, G: float = 0, N: float = 0,
            kernel: collections.abc.Callable[[float, float, float], float] = 0, beta: float = 0,
            state_diff: ca.SX = None, tau=1) -> ca.SX:
        '''
        Compute the right-hand side of the PBE for the given PBE method.

        Parameters:
        - states: Current states of the PBE
        - alg_states: Current algebraic states of the PBE
        - G: Growth rate
        - N: Nuclation rate
        - kernel: Kernel function for agglomeration
        - beta: Agglomeration parameter
        - Dil: Dilution rate
        - state_in: Values of states for inflowing stream
        '''

        # setup rhs function with G, N, kernel, beta as inputs, incoming and outcoming flux for PBE state
        if self.method == 'SMOM':
            if state_diff is None:
                state_diff = ca.SX.zeros(self.n_moments)
            dot_mu = ca.SX.zeros(self.n_moments)
            dot_mu[0] = N * tau + state_diff[0]
            for k in range(1, self.n_moments):
                dot_mu[k] = k * G * states[k - 1] * tau + state_diff[k]

            return dot_mu

        elif self.method == 'QMOM':
            if state_diff is None:
                state_diff = ca.SX.zeros(6)
            # PD algorithm
            L = ca.SX.zeros(3)
            w = ca.SX.zeros(3)

            J3 = self.J_fun(states / states[0])  # scale by zeroth moment
            eigval = ca.eig_symbolic(J3)
            L = eigval
            eigvec = self.eigvec_fun(states / states[0], eigval)
            w = states[0] * (eigvec ** 2)  # unscale by zeroth moment

            dot_mu = ca.SX.zeros(6)

            dot_mu[0] = N * tau + state_diff[0] + self.agg_fun(L, w, 0, beta) * tau  # +break_sum(L, w, a, b, 0, beta_a)
            for k in range(1, 6):
                dot_mu[k] = k * G * (ca.sum1(L ** (k - 1) * w)) * tau + state_diff[k] + self.agg_fun(L, w, k,
                                                                                                     beta) * tau  # +break_sum(L, w, a, b, k, beta_a)

            return dot_mu

        elif self.method == 'DPBE':
            states = ca.reshape(states, -1, 1)
            if state_diff is None:
                state_diff = ca.SX.zeros(self.no_class, 1)
            else:
                state_diff = ca.reshape(state_diff, -1, 1)
            # calculate agglomeration matrix for all possible collisions Litster
            beta_Agg = ca.SX.zeros(self.no_class, self.no_class)
            dn_Agg_dt = ca.SX.zeros(self.no_class)
            B_agg = ca.SX.zeros(self.no_class)
            D_agg = ca.SX.zeros(self.no_class)

            if isinstance(beta, (ca.SX, ca.MX)) or beta != 0:
                print('Calc agglomeration')
                if self.spacing == 'geometric':
                    # function needed for calculation by Litster
                    S = lambda q: np.sum([np.arange(q + 1)])
                    for i in range(self.no_class):
                        for j in range(self.no_class):
                            beta_Agg[i, j] = kernel(self.L_i[i], self.L_i[j], beta)

                    N_i = states * self.del_L_i
                    for i in range(self.no_class):
                        B = 0
                        D = 0
                        for j in range(i - S(self.q)):
                            B += beta_Agg[i - 1, j] * N_i[i - 1] * N_i[j] * (2 ** ((j - i + 1) / self.q)) / (
                                    2 ** (1 / self.q) - 1)
                        if i - self.q >= 0:
                            B += 0.5 * beta_Agg[i - self.q, i - self.q] * N_i[i - self.q] ** 2
                        for k in range(2, self.q + 1):
                            for j in range(max(0, i - S(self.q - k + 2) - k + 1), i - S(self.q - k + 1) - k + 1):
                                B += beta_Agg[i - k, j] * N_i[i - k] * N_i[j] * (
                                        2 ** ((j - i + 1) / self.q) - 1 + 2 ** (-(k - 1) / self.q)) / (
                                             2 ** (1 / self.q) - 1)
                            for j in range(max(0, i - S(self.q - k + 2) - k + 2), i - S(self.q - k + 1) - k + 2):
                                B += beta_Agg[i - k + 1, j] * N_i[i - k + 1] * N_i[j] * (
                                        -2 ** ((j - i) / self.q) + 2 ** (1 / self.q) - 2 ** (
                                        -(k - 1) / self.q)) / (2 ** (1 / self.q) - 1)
                        for j in range(1, i - S(self.q) + 1):
                            D += beta_Agg[i, j] * N_i[i] * N_i[j] * (2 ** ((j - i) / self.q)) / (
                                    2 ** (1 / self.q) - 1)
                        for j in range(max(0, i - S(self.q) + 1), self.no_class):
                            D += beta_Agg[i, j] * N_i[i] * N_i[j]
                        B_agg[i] = B
                        D_agg[i] = D
                        dn_Agg_dt[i] = (B - D) / self.del_L_i[i]

                # agglomeration for uniform spacing
                elif self.spacing == 'uniform':

                    if self.coordinate == 'L':
                        raise ValueError(
                            'Agglomeration currently only implemented for geometric spacing for L as inner coordinate.')

                        # N_i = states*self.del_L_i
                        # for i in range(self.no_class):
                        #     for j in range(i):
                        #         # B_agg[i] += 0.5*(self.L_i[i]**2)*kernel(self.L_i[j],self.L_i[i-j],beta)*((N_i[self.map[i,j]+1]-N_i[self.map[i,j]])/
                        #         #                                                     (self.L_i[self.map[i,j]+1]-self.L_i[self.map[i,j]])*
                        #         #                                                     ((self.L_i[i]**3-self.L_i[j]**3)**(1/3)-self.L_i[self.map[i,j]])+N_i[self.map[i,j]])*N_i[j]/((self.L_i[i]**3-self.L_i[j]**3)**(2/3)) # implementation using interpolation, doesn't work for this implementation because of counter i+1
                        #         B_agg[i] += 0.5*(self.L_i[i]**2)*kernel(self.L_i[j],self.L_i[i-j],beta)*N_i[self.map[i,j]]*N_i[j]/((self.L_i[i]**3-self.L_i[j]**3)**(2/3)) # substracted 1 everywhere from i
                        #     for j in range(self.no_class):
                        #         D_agg[i] += kernel(self.L_i[i],self.L_i[j],beta)*N_i[j]
                        #     D_agg[i] = D_agg[i]*N_i[i]
                        #     dn_Agg_dt[i] = (B_agg[i]-D_agg[i])/self.del_L_i[i]

                    elif self.coordinate == 'V':
                        N_i = states * self.del_L_i
                        for i in range(self.no_class):
                            for j in range(i):
                                B_agg[i] += 0.5 * kernel(self.L_i[j], self.L_i[i - j - 1], beta) * N_i[j] * N_i[
                                    i - j - 1]
                            for j in range(self.no_class):
                                D_agg[i] += kernel(self.L_i[i], self.L_i[j], beta) * N_i[j]
                            D_agg[i] = D_agg[i] * N_i[i]
                            dn_Agg_dt[i] = (B_agg[i] - D_agg[i]) / self.del_L_i[i]
                    else:
                        raise ValueError('Invalid coordinate. Please choose from L or V.')

            dn_dt = ca.SX.zeros(self.no_class)
            dn_dt_G = ca.SX.zeros(self.no_class)
            r = ca.SX.zeros(self.no_class)
            phi_r = ca.SX.zeros(self.no_class)
            Gn_Li = ca.SX.zeros(self.no_class)
            eps = 1e-10
            for i in range(1, self.no_class - 1):
                r[i] = (states[i + 1] - states[i] + eps) / (states[i] - states[i - 1] + eps)
                phi_r[i] = ca.fmax(0, ca.fmin(2 * r[i], ca.fmin(1 / 3 + 2 / 3 * r[i], 2)))

            # G is growth function which is evaluated at the boundaries
            Gn_Li = ca.SX.zeros(self.no_class + 1)
            Gn_Li = ca.SX.zeros(self.no_class + 1)

            # setup growth function for specific time
            # G_fun_0 is growth function with unit rate
            G_fun = lambda L: self.G_fun_0(L) * G

            # First order
            if self.scheme == 'first':
                G_faces = ca.vertcat(*[G_fun(boundary) for boundary in self.L_i_bound])
                G_faces_pos = ca.fmax(G_faces, 0)
                G_faces_neg = ca.fmin(G_faces, 0)

                Gn_Li[0] = N + G_faces_neg[0] * states[0]
                Gn_Li[1:-1] = (
                    G_faces_pos[1:-1] * states[:-1]
                    + G_faces_neg[1:-1] * states[1:]
                )
                Gn_Li[-1] = G_faces_pos[-1] * states[-1]

            # Second order
            elif self.scheme == 'second':
                NG = N / ca.fmax(1e-10, G_fun(0))
                Gn_Li[0] = NG
                Gn_Li[1:-1] = G_fun(self.L_i[1:] - self.del_L_i[1:] / 2) * 0.5 * (states[:-1] + states[1:])
                Gn_Li[-1] = G_fun(self.L_i[-1] + self.del_L_i[-1] / 2) * states[-1]

            # limited k scheme with k = 1/3 using upwind ratio r
            elif self.scheme == 'limited':
                NG = N / ca.fmax(1e-10, G_fun(0))
                Gn_Li[0] = NG
                Gn_Li[1] = G_fun(self.L_i[1] - self.del_L_i[1] / 2) * states[0]
                Gn_Li[2:-2] = G_fun(self.L_i[2:-1] - self.del_L_i[2:-1] / 2) * (
                        states[1:-2] + 0.5 * phi_r[1:-2] * (states[1:-2] - states[0:-3]))
                Gn_Li[-2] = G_fun(self.L_i[-2] - self.del_L_i[-1] / 2) * states[-2]
                Gn_Li[-1] = G_fun(self.L_i[-1] + self.del_L_i[-1] / 2) * states[-1]

                # WENO
            elif self.scheme == 'WENO5':
                NG = N / ca.fmax(1e-10, G_fun(0))
                Gn_Li = functions.WENO5(states, NG)
                Gn_Li[1:] = G_fun(self.L_i_bound[1:]) * Gn_Li[
                                                        1:]  # no need to multiply with boundary condition --> [1:]
            elif self.scheme == 'WENO7':
                NG = N / ca.fmax(1e-10, G_fun(0))
                Gn_Li = functions.WENO7(states, NG)
                Gn_Li[1:] = G_fun(self.L_i_bound[1:]) * Gn_Li[1:]

            dn_dt_G = -1 / self.del_L_i * (Gn_Li[1:] - Gn_Li[:-1])

            dn_dt[0] = dn_Agg_dt[0] * tau + dn_dt_G[0] * tau + state_diff[0]
            dn_dt[1] = dn_Agg_dt[1] * tau + dn_dt_G[1] * tau + state_diff[1]
            dn_dt[2:-1] = dn_Agg_dt[2:-1] * tau + dn_dt_G[2:-1] * tau + state_diff[2:-1]
            dn_dt[-1] = dn_Agg_dt[-1] * tau + dn_dt_G[-1] * tau + state_diff[-1]

            return dn_dt

        elif self.method == 'OCFE':
            states = ca.reshape(states, self.n_elements, self.n_col - 2)
            alg_states = ca.reshape(alg_states, self.n_elements, 2)

            if state_diff is None:
                state_diff = ca.SX.zeros(self.n_elements, self.n_col - 2)
            else:
                state_diff = ca.reshape(state_diff, self.n_elements, self.n_col - 2)

            n_full = ca.SX.zeros(self.n_elements, self.n_col)
            for element in range(self.n_elements):
                n_full[element, :] = ca.horzcat(alg_states[element, 0], states[element, :], alg_states[element, 1])

            # jacobian of transformation for current number of elements to [0,1]:
            J = self.step
            self.J_D = J

            # agglomeration
            D = ca.SX.zeros(self.n_elements, self.n_col - 2)
            B = ca.SX.zeros(self.n_elements, self.n_col - 2)
            B1 = ca.SX.zeros(self.n_elements, self.n_col - 2)
            B2 = ca.SX.zeros(self.n_elements, self.n_col - 2)
            if kernel is not None:
                for element in range(self.n_elements):
                    for col in range(self.n_col - 2):
                        # V_i_e is current collocation point
                        V_i_e = self.n_pos[element, col]

                        # death rate due to agglomeration
                        for e in range(self.n_elements):
                            D[element, col] += self.J_D[e] * np.sum(
                                [self.weights[k] * kernel(V_i_e, self.n_pos[e, k], beta) * states[e, k] for k in range(
                                    self.n_col - 2)])  # range 2 should later include all collocation points, must adapt special quadrature formula
                        D[element, col] = D[element, col] * states[element, col]

                        if self.coordinate == 'L':
                            # birth due to agglomeration for L as internal coordinate
                            # calculation from Alexopoulos must be adapted to L
                            # B1:
                            # calculate contribution to L_i_e up to element g-1 (only part of element g lies within (L_i_e**3)/2)
                            # g is element which contains V_i_e/2
                            g = self.map2[element, col]
                            for e in range(g):
                                for k in range(self.n_col - 2):
                                    # find element h which contains (V_i_e-U) --> map1
                                    # U is n_pos[e,k]
                                    h = self.map1[
                                        element, col, e, k + 1]  # (k+1) necessary because k goes to n_col-2 but map1 is defined for n_col

                                    # once h is known V_i_e-U must be scaled to the local domain within h to evaluate approximating polynomial
                                    v_i = ((V_i_e ** 3 - self.n_pos[e, k] ** 3) ** (1 / 3) - self.n_pos_full[h, 0]) / \
                                          self.step[h]
                                    v_bar = np.array([v_i ** i for i in range(self.n_col)])
                                    B1[element, col] += J[e] * self.weights[k] * kernel(V_i_e - self.n_pos[e, k],
                                                                                        self.n_pos[e, k], beta) * \
                                                        states[e, k] * n_full[h, :] @ self.coef @ v_bar / ((self.n_pos[
                                                                                                                element, col] ** 3 -
                                                                                                            self.n_pos[
                                                                                                                e, k] ** 3) ** (
                                                                                                                   2 / 3))

                            # B2:
                            # e is now g
                            # calculate rest of contribution of element g for V_i_e ([element,col])
                            # hard coded for two collocation points plus bounds

                            # find position of V_g within g to determine necessary number of loops for element g
                            V_g = V_i_e / (2 ** (1 / 3))
                            for m in reversed(range(self.n_col - 1)):
                                if V_g >= self.n_pos_full[g, m]:
                                    for k in reversed(range(m + 1)):
                                        if k < m:
                                            # determine V_p for k
                                            V_p = self.n_pos_full[g, k + 1]

                                            # both bounds are collocation points --> calculate v_bar_0 and v_bar_1
                                            el_0 = self.map1[element, col, g, k]
                                            v_i_0 = ((V_i_e ** 3 - self.n_pos_full[g, k] ** 3) ** (1 / 3) -
                                                     self.n_pos_full[el_0, 0]) / self.step[el_0]
                                            v_bar_0 = np.array([v_i_0 ** i for i in range(self.n_col)])
                                            # evaluation of n at v_i_0
                                            n_0 = n_full[el_0, :] @ self.coef @ v_bar_0

                                            el_1 = self.map1[element, col, g, k + 1]
                                            v_i_1 = ((V_i_e ** 3 - self.n_pos_full[g, k + 1] ** 3) ** (1 / 3) -
                                                     self.n_pos_full[el_1, 0]) / self.step[el_1]
                                            v_bar_1 = np.array([v_i_1 ** i for i in range(self.n_col)])
                                            # evaluation of n at v_i_1
                                            n_1 = n_full[el_1, :] @ self.coef @ v_bar_1

                                            B2[element, col] += 0.5 / self.step[g] * J[g] * (
                                                    V_p - self.n_pos_full[g, k]) * (
                                                                        kernel(V_i_e - self.n_pos_full[g, k],
                                                                               self.n_pos_full[g, k], beta) * n_0 *
                                                                        n_full[g, k] +
                                                                        kernel(V_i_e - V_p, V_p, beta) * n_1 *
                                                                        n_full[g, k + 1]) / ((self.n_pos[
                                                                                                  element, col] ** 3 -
                                                                                              self.n_pos_full[
                                                                                                  g, k] ** 3) ** (
                                                                                                     2 / 3))

                                        else:
                                            V_p = V_g

                                            # both bounds are collocation points --> calculate v_bar_0 and v_bar_1
                                            # additionally we need (V_i_e-V_g)  because right bound is not collocation point
                                            el_0 = self.map1[element, col, g, k]
                                            v_i_0 = ((V_i_e ** 3 - self.n_pos_full[g, k] ** 3) ** (1 / 3) -
                                                     self.n_pos_full[el_0, 0]) / self.step[el_0]
                                            v_bar_0 = np.array([v_i_0 ** i for i in range(self.n_col)])
                                            n_0 = n_full[el_0, :] @ self.coef @ v_bar_0

                                            el_p = self.map2[element, col]
                                            v_i_p = ((V_i_e ** 3 - V_g ** 3) ** (1 / 3) - self.n_pos_full[el_p, 0]) / \
                                                    self.step[el_p]
                                            v_bar_p = np.array([v_i_p ** i for i in range(self.n_col)])
                                            n_p = n_full[el_p, :] @ self.coef @ v_bar_p

                                            B2[element, col] += 0.5 / self.step[g] * J[g] * (
                                                    V_p - self.n_pos_full[g, k]) * (
                                                                        kernel(V_i_e - self.n_pos_full[g, k],
                                                                               self.n_pos_full[g, k], beta) * n_0 *
                                                                        n_full[g, k] +
                                                                        kernel(V_i_e - V_p, V_p,
                                                                               beta) * n_p * n_p) / ((self.n_pos[
                                                                                                          element, col] ** 3 -
                                                                                                      self.n_pos_full[
                                                                                                          g, k] ** 3) ** (
                                                                                                             2 / 3))
                                    break
                            B[element, col] = (self.n_pos[element, col] ** 2) * (B1[element, col] + B2[element, col])

                        elif self.coordinate == 'V':
                            # birth due to agglomeration
                            # B1:
                            # calculate contribution to V_i_e up to element g-1 (only part of element g lies within V_i_e/2)
                            # g is element which contains V_i_e/2
                            g = self.map2[element, col]
                            for e in range(g):
                                for k in range(self.n_col - 2):
                                    # find element h which contains (V_i_e-U) --> map1
                                    # U is n_pos[e,k]
                                    h = self.map1[
                                        element, col, e, k + 1]  # (k+1) necessary because k goes to n_col-2 but map1 is defined for n_col

                                    # once h is known V_i_e-U must be scaled to the local domain within h to evaluate approximating polynomial
                                    v_i = ((V_i_e - self.n_pos[e, k]) - self.n_pos_full[h, 0]) / self.step[h]
                                    v_bar = np.array([v_i ** i for i in range(self.n_col)])
                                    B1[element, col] += J[e] * self.weights[k] * kernel(V_i_e - self.n_pos[e, k],
                                                                                        self.n_pos[e, k], beta) * \
                                                        states[e, k] * n_full[h, :] @ self.coef @ v_bar
                                    # h correct in n_full[h,:] above????
                            # B2:
                            # e is now g
                            # calculate rest of contribution of element g for V_i_e ([element,col])
                            # hard coded for two collocation points plus bounds

                            # find position of V_g within g to determine necessary number of loops for element g
                            V_g = V_i_e / 2
                            for m in reversed(range(self.n_col - 1)):
                                if V_g >= self.n_pos_full[g, m]:
                                    for k in reversed(range(m + 1)):
                                        if k < m:
                                            # determine V_p for k
                                            V_p = self.n_pos_full[g, k + 1]

                                            # both bounds are collocation points --> calculate v_bar_0 and v_bar_1
                                            el_0 = self.map1[element, col, g, k]
                                            v_i_0 = ((V_i_e - self.n_pos_full[g, k]) - self.n_pos_full[el_0, 0]) / \
                                                    self.step[el_0]
                                            v_bar_0 = np.array([v_i_0 ** i for i in range(self.n_col)])
                                            # evaluation of n at v_i_0
                                            n_0 = n_full[el_0, :] @ self.coef @ v_bar_0

                                            el_1 = self.map1[element, col, g, k + 1]
                                            v_i_1 = ((V_i_e - self.n_pos_full[g, k + 1]) - self.n_pos_full[el_1, 0]) / \
                                                    self.step[el_1]
                                            v_bar_1 = np.array([v_i_1 ** i for i in range(self.n_col)])
                                            # evaluation of n at v_i_1
                                            n_1 = n_full[el_1, :] @ self.coef @ v_bar_1

                                            B2[element, col] += 0.5 / self.step[g] * J[g] * (
                                                    V_p - self.n_pos_full[g, k]) * (
                                                                        kernel(V_i_e - self.n_pos_full[g, k],
                                                                               self.n_pos_full[g, k], beta) * n_0 *
                                                                        n_full[g, k] +
                                                                        kernel(V_i_e - V_p, V_p, beta) * n_1 *
                                                                        n_full[g, k + 1])

                                        else:
                                            V_p = V_g

                                            # both bounds are collocation points --> calculate v_bar_0 and v_bar_1
                                            # additionally we need (V_i_e-V_g)  because right bound is not collocation point
                                            el_0 = self.map1[element, col, g, k]
                                            v_i_0 = ((V_i_e - self.n_pos_full[g, k]) - self.n_pos_full[el_0, 0]) / \
                                                    self.step[el_0]
                                            v_bar_0 = np.array([v_i_0 ** i for i in range(self.n_col)])
                                            n_0 = n_full[el_0, :] @ self.coef @ v_bar_0

                                            el_p = self.map2[element, col]
                                            v_i_p = ((V_i_e - V_g) - self.n_pos_full[el_p, 0]) / self.step[el_p]
                                            v_bar_p = np.array([v_i_p ** i for i in range(self.n_col)])
                                            n_p = n_full[el_p, :] @ self.coef @ v_bar_p

                                            B2[element, col] += 0.5 / self.step[g] * J[g] * (
                                                    V_p - self.n_pos_full[g, k]) * (
                                                                        kernel(V_i_e - self.n_pos_full[g, k],
                                                                               self.n_pos_full[g, k], beta) * n_0 *
                                                                        n_full[g, k] +
                                                                        kernel(V_i_e - V_p, V_p, beta) * n_p * n_p)
                                    break

            if self.coordinate == 'V':
                B = B1 + B2

            # growth
            # setup growth function for specific time
            # G_fun_0 is grwoth function with unit rate
            self.G_fun = lambda L: self.G_fun_0(L) * G
            self.G_fun_deriv = lambda L: self.G_fun_0_deriv(L) * G

            G_i = ca.SX.zeros(self.n_elements, self.n_col - 2)
            for element in range(self.n_elements):
                for col in range(self.n_col - 2):
                    G_i[element, col] = 1 / J[element] * self.G_fun(self.n_pos[element, col]) * ca.reshape(
                        self.s[col + 1, :], 1, -1) @ ca.reshape(n_full[element, :], -1, 1) + self.G_fun_deriv(
                        self.n_pos[element, col]) * states[element, col]

            # artificial diffusion term
            Diff = ca.SX.zeros(self.n_elements, self.n_col - 2)
            for element in range(self.n_elements):
                for col in range(self.n_col - 2):
                    Diff[element, col] = self.D_a * 1 / J[element] ** 2 * ca.reshape(self.s_2[col + 1, :], 1,
                                                                                     -1) @ ca.reshape(
                        n_full[element, :], -1, 1)

            # rhs
            dot_n = ca.SX.zeros(self.n_elements, self.n_col - 2)
            for element in range(self.n_elements):
                for col in range(self.n_col - 2):
                    dot_n[element, col] = -G_i[element, col] + Diff[element, col] - D[element, col] + B[element, col] + \
                                          state_diff[element, col] / tau
            # dot_n[0,0] += N

            # dot_n[0,0] += N/(self.n_pos[0,1]-self.n_pos[0,0])
            # dot_n[0,1] += N/(self.n_pos[0,1]-self.n_pos[0,0])

            return ca.reshape(dot_n * tau, -1, 1)
        else:
            raise ValueError('Invalid PBE method. Please choose from SMOM, QMOM, DPBE, or OCFE.')

    def alg(self, states: ca.SX, alg_states: ca.SX, aux_state: ca.SX, G: float = 0, N: float = 0,
            kernel: collections.abc.Callable[[float, float, float], float] = 0, beta: float = 0,
            state_diff: ca.SX = None):
        '''
        Compute the algebraic equations for the given PBE method.

        Parameters:
        - states: Current states of the PBE
        - alg_states: Current algebraic states of the PBE
        - G: Growth rate
        - N: Nuclation rate
        - kernel: Kernel function for agglomeration
        - beta: Agglomeration parameter
        '''
        if self.method == 'SMOM':
            return None
        elif self.method == 'QMOM':
            pass
        elif self.method == 'DPBE':
            pass
        elif self.method == 'OCFE':
            states = ca.reshape(states, self.n_elements, self.n_col - 2)
            alg_states = ca.reshape(alg_states, self.n_elements, 2)

            if state_diff is None:
                state_diff = ca.SX.zeros(self.n_elements, self.n_col - 2)

            n_full = ca.SX.zeros(self.n_elements, self.n_col)
            for element in range(self.n_elements):
                n_full[element, :] = ca.horzcat(alg_states[element, 0], states[element, :], alg_states[element, 1])

            # compute rhs for auxillary state for boundary condition
            boundary_rhs = ca.SX(0)
            if self.boundary_condition == 'agglomeration':
                for e in range(self.n_elements):
                    boundary_rhs += self.J_D[e] * np.sum(
                        [self.weights[k] * kernel(0, self.n_pos[e, k], beta) * states[e, k] for k in range(
                            self.n_col - 2)])  # range 2 should later include all collocation points, must adapt special quadrature formula
                boundary_rhs = -boundary_rhs * n_full[0, 0] + state_diff[0, 0]  # not sure about state_diff
            elif self.boundary_condition == 'nucleation':
                boundary_rhs = N / self.G_fun(self.n_pos_full[0, 0]) - self.G_fun(self.n_pos_full[0, 0]) * n_full[
                    0, 0] + state_diff[0, 0]

            alg = ca.SX.zeros(self.n_elements * 2)

            # alg[0] = ca.reshape(self.s[0,:],1,-1)@ca.reshape(n_full[0,:],-1,1) # enforces derivate of n at first collocation point to be 0
            # alg[0] = ca.reshape(self.s_2[0,:],1,-1)@ca.reshape(n_full[0,:],-1,1) # enforces second derivate of n at first collocation point to be 0
            # alg[0] = ca.reshape(self.s_3[0,:],1,-1)@ca.reshape(n_full[0,:],-1,1) # enforces third derivate of n at first collocation point to be 0
            # alg[0] = ca.reshape(self.s_4[0,:],1,-1)@ca.reshape(n_full[0,:],-1,1) # enforces fourth derivate of n at first collocation point to be 0

            # alg[0] = alg_states[0,0]-N/(G+1e-10)#ca.reshape(self.s[0,:],1,-1)@ca.reshape(n_full[0,:],-1,1)#alg_states[0,0]#-N/(G+1e-10)

            # select boundary condition
            if self.boundary_condition == 'zero':
                alg[0] = alg_states[0, 0]
            elif self.boundary_condition == 'nucleation':
                # alg[0] = alg_states[0,0]-N/G
                alg[0] = alg_states[0, 0] - aux_state
            elif self.boundary_condition == 'agglomeration':
                alg[0] = alg_states[0, 0] - aux_state

            alg[1] = ca.reshape(self.s[-1, :], 1, -1) @ ca.reshape(n_full[-1, :], -1, 1)
            # alg[1] = alg_states[-1,-1]-1e-20

            # enforce connection of elements
            for element in range(self.n_elements - 1):
                alg[element + 2] = alg_states[element, 1] - alg_states[element + 1, 0]

            # continuity of f'
            for element in range(self.n_elements - 1):
                alg[element + 1 + self.n_elements] = ca.reshape(self.s[-1, :], 1, -1) @ ca.reshape(n_full[element, :],
                                                                                                   -1, 1) - ca.reshape(
                    self.s[0, :], 1, -1) @ ca.reshape(n_full[element + 1, :], -1, 1)
            test = alg[0]
            return alg, boundary_rhs
        else:
            raise ValueError('Invalid PBE method. Please choose from SMOM, QMOM, DPBE, or OCFE.')

    def calc_moments(self, states: ca.SX, alg_states: ca.SX) -> ca.SX:
        '''
        Calculate moments from PSD.

        Parameters:
        - states: Current states of the PBE
        - alg_states: Current algebraic states of the PBE
        '''
        if self.method == 'SMOM':
            return states
        elif self.method == 'QMOM':
            return states
        elif self.method == 'DPBE':
            return np.array([ca.sum1(ca.reshape(states, -1, 1) * self.del_L_i * self.L_i ** k) for k in range(6)])
        elif self.method == 'OCFE':
            states = ca.reshape(states, self.n_elements, self.n_col - 2)
            alg_states = ca.reshape(alg_states, self.n_elements, 2)

            # calculate moments for OCFE
            # lagrange polynomials are evaluated at different points in each element
            # higher number of evaluation points per element leads to better approximation of the integral but also to higher computational cost
            n_xx = 100  # number of evaluation points per element

            n_full = ca.SX.zeros(self.n_elements, self.n_col)
            for element in range(self.n_elements):
                n_full[element, :] = ca.horzcat(alg_states[element, 0], states[element, :], alg_states[element, 1])

            n_moments = 6
            mu = ca.SX.zeros(n_moments)
            for el in range(self.n_elements):
                domain = [self.n_pos_full[el, 0], self.n_pos_full[el, -1]]
                xx = np.linspace(*domain, n_xx)
                for x in xx:
                    mu += (domain[1] - domain[0]) / n_xx * np.array([(n_full[el, :] @ self.coef @ np.array(
                        [((x - self.n_pos_full[el, 0]) / (self.n_pos_full[el, -1] - self.n_pos_full[el, 0])) ** i for i
                         in range(self.n_col)]).reshape(-1, 1)) * x ** i for i in range(n_moments)]).ravel()
            return mu
        else:
            raise ValueError('Invalid PBE method. Please choose from SMOM, QMOM, DPBE, or OCFE.')


def eval_OCFE(xx, metadata_OCFE, results_OCFE, t=-1):
    '''
    Evaluate the OCFE method at given points.

    Parameters:
    - xx: Points to evaluate the OCFE method
    - PBE: PBE object
    - simulator_obj: Simulator object
    - t: Time to evaluate the OCFE method (default: last time step)
    '''
    # if t ==-1:
    n_elements = metadata_OCFE['kwargs']['n_elements']
    n_col = metadata_OCFE['kwargs']['n_col']

    n_pos_full = metadata_OCFE['n_pos_full']
    coef = metadata_OCFE['coef']

    n_alg = results_OCFE['simulator']['_z', 'PBE_alg'][t, :]
    n_x = np.array(ca.reshape(results_OCFE['simulator']['_x', 'PBE_state'][t, :], n_elements, (n_col - 2)))
    # else:
    #     n_alg = simulator_obj.data._z[t+1,:]
    #     n_x = np.array(ca.reshape(simulator_obj.data._x[t+1,:][:-1],PBE.n_elements,(PBE.n_col-2)))

    n_full = np.concatenate(
        [np.concatenate((n_alg[k].reshape(1, -1), n_x[k, :].reshape(1, -1), n_alg[k + 1].reshape(1, -1)), axis=1) for k
         in range(n_elements)], axis=0)
    n_full[-1, -1] = n_alg[-1]  # correct last element

    warning = True
    n_out = []
    for x_i in np.array(xx).reshape(-1):
        for element in range(n_elements):
            if x_i >= n_pos_full[element, 0] and x_i <= n_pos_full[element, -1]:
                x_i_sc = ((x_i - n_pos_full[element, 0]) / (n_pos_full[element, -1] - n_pos_full[element, 0]))
                x_eval = np.array([x_i_sc ** k for k in range(n_col)])
                n_out.append(n_full[element, :] @ coef @ x_eval)
                break
            elif x_i < n_pos_full[0, 0]:
                if warning:
                    print('Warning: x_i is smaller than smallest collocation point. Using first element.')
                    warning = False
                x_i_sc = ((x_i - n_pos_full[0, 0]) / (n_pos_full[0, -1] - n_pos_full[0, 0]))
                x_eval = np.array([x_i_sc ** k for k in range(n_col)])
                n_out.append(n_full[0, :] @ coef @ x_eval)
                break
            elif x_i > n_pos_full[-1, -1]:
                if warning:
                    print('Warning: x_i is larger than largest collocation point. Using last element.')
                    warning = False
                x_i_sc = ((x_i - n_pos_full[-1, 0]) / (n_pos_full[-1, -1] - n_pos_full[-1, 0]))
                x_eval = np.array([x_i_sc ** k for k in range(n_col)])
                n_out.append(n_full[-1, :] @ coef @ x_eval)
                break
    return np.array(n_out)


class MC_simulation:
    '''
    Class for solution of PBE by Monte Carlo simulation.

    Can cover the following phenomena: size-independent growth, size-dependent growth,  and constant agglomeration.
    Implementation only for single element used for comparison to other solution methods.

    Parameters:

    Methods:
    - __init__: Initialize the Monte Carlo simulation.
        Provide:    - n_init: initial distribution (must be a callable function)
                    - G: growth rate
                    - B: nucleation rate
                    - beta: agglomeration parameter
                    - Dil: dilution rate
                    - n_init: initial distribution
                    - domain: domain of the distribution
                    - max: maximum of initial distribution within domain (optional for more accurate sampling)

    - init_particles: Initialize the particles for the Monte Carlo simulation.
        Provide:    - n_particles: number of particles to simulate

    - simulate: Perform Monte Carlo simulation.
        Provide:    - t: time to simulate
                    - dt: time step for simulation (default: 0.1) (optional)
                    - silence: suppress output (optional)

    Information about method plot: normalized: bool = True, bins: int = 100, func: collections.abc.Callable[[float], float] = None):

    - plot: Plot the current particle distribution.
        Provide:    - normalized: normalize the histogram (optional)
                    - bins: number of bins for histogram (optional)
                    - func: function to plot over histogram (e.g. an intial distribution or analytic solution) (optional)
        '''

    def __init__(self, n_init, G: float, B: float, beta: float, Dil: float,
                 domain: list, coordinate: str, max: float = 0) -> None:
        '''
        Initialize the Monte Carlo simulation.

        Parameters:
        - n_init: initial distribution (must be a callable function)
        - G: growth rate
        - B: nucleation rate
        - beta: agglomeration parameter
        - Dil: dilution rate
        - domain: domain of the distribution
        - max: maximum of initial distribution within domain (optional for more accurate sampling)
        '''
        self.PBE = PBE
        self.n_init = n_init
        self.G = G
        self.B = B
        self.beta = beta
        self.Dil = Dil
        self.mu = None
        ###################################################
        self.m_0 = 1  # what is m_0? #######################
        ###################################################
        self.coordinate = coordinate
        self.domain = domain
        self.cut_off_agg = 0
        self.cut_off_nuc = 0
        if max:
            self.max = max
        else:
            self.max = np.max(n_init(np.linspace(*domain, 10000)))

    def init_particles(self, n_particles: int = None, mu_3: float = None, silence: bool = False) -> None:
        '''
        Initialize the particles for the Monte Carlo simulation.

        Sample n_particles from the initial distribution by rejection method.
        Or sample n_particles until certain mu_3 is reached.

        Parameters:
        - n_particles: number of particles to simulate

        - silence: suppress output
        '''
        self.n_particles = n_particles

        # particle list
        self.particles = []

        # sample particles
        counter = 0
        
        if n_particles:
            while counter < self.n_particles:
                # generate random point within rectangle
                r1 = np.random.uniform()
                r2 = np.random.uniform()

                # scale to domain
                r1_sc = r1 * (self.domain[1] - self.domain[0]) + self.domain[0]
                r2_sc = r2 * self.max

                # accept if below pdf
                if self.n_init(r1_sc) > r2_sc:
                    self.particles.append(r1_sc)
                    counter += 1
        elif mu_3:
            while True:
                # generate random point within rectangle
                r1 = np.random.uniform()
                r2 = np.random.uniform()

                # scale to domain
                r1_sc = r1 * (self.domain[1] - self.domain[0]) + self.domain[0]
                r2_sc = r2 * self.max

                # accept if below pdf
                if self.n_init(r1_sc) > r2_sc:
                    self.particles.append(r1_sc)
                    counter += 1

                    # compute moments
                    self.particles = np.array(self.particles)
                    self.mu = np.array([np.sum(self.particles ** k) for k in range(6)])
                    self.particles = self.particles.tolist()

                    if self.mu[3] > mu_3:
                        break
        else:
            raise ValueError('Please provide either number of particles or mu_3.')

        self.particles = np.array(self.particles)
        self.d10 = np.percentile(self.particles, 10)
        self.d50 = np.percentile(self.particles, 50)
        self.d90 = np.percentile(self.particles, 90)

        if not (silence):
            print('-----------------------------------')
            print(f'Initial distribution:')
            print(f'Number of particles: {self.particles.shape[0]}')
            print(f'10th percentile (d10): {self.d10}')
            print(f'50th percentile (d50): {self.d50}')
            print(f'90th percentile (d90): {self.d90}')
            print(f'Width of distribution: {self.d90 - self.d10}')
            print('-----------------------------------')

        # compute moments
        self.mu = np.array([np.sum(self.particles ** k) for k in range(6)])

    def simulate(self, t: float, dt: float = 0.1, silence: bool = False,
                 G_fun_0: collections.abc.Callable[[float], float] = None, return_full_simulation: bool = False,
                 G: float = None, B: float = None, beta: float = None, Dil: float = None, kernel_function = None) -> None:
        '''
        Make a step in the Monte Carlo simulation.

        Simulate change of particles due to growth, nucleation, aggregation, and dilution.

        Parameters:
        - t: total simulation time
        - dt: time step for simulation (default: 0.1)
        - silence: suppress output
        - G_fun: growth function (optional) (default: size-independent growth)
        '''

        # check if new parameters are provided, otherwise use old ones
        if G:
            self.G = G
        if B:
            self.B = B
        if beta:
            self.beta = beta
        if Dil:
            self.Dil = Dil

        self.t = t
        self.dt = dt

        steps = int(t / dt)

        # initialize list for each time step
        particle_list = []

        # add initial distribution to list
        particle_list.append(self.particles.reshape(1, -1))

        # G_fun_0 is growth function with unit rate
        if G_fun_0:
            G_fun = lambda L: G_fun_0(L) * self.G

        # get some information about particle distribution before simulation
        no_particles_start = self.particles.shape[0]

        time_start = time.perf_counter()
        for t in range(steps):
            # number of simulated samples
            PopnNo = self.particles.shape[0]

            # calculate crystal growth
            if G_fun_0:
                # size-dependent growth if growth function provided
                self.particles = self.particles + G_fun(self.particles) * dt
            else:
                # size-independent growth
                self.particles = self.particles + self.G * dt

            # for sample in range(PopnNo):
            #     # currently only constant growth implemented
            #     pass

            # calculate number of agglomeration events for given time-step
            m_0_next = self.m_0 - 0.5 * self.beta * self.m_0 ** 2 * dt
            delta_m_0 = self.m_0 - m_0_next

            # can only simulate integer number of agglomeration events
            # --> round-off error is accumulated
            AggNo = PopnNo * delta_m_0 / self.m_0 + self.cut_off_agg

            self.m_0 = m_0_next
            self.cut_off_agg = AggNo - int(AggNo)
            AggNo = int(AggNo)
            
            # print(f'{AggNo} agglomeration events for a population of {PopnNo} particles')
            

            # perform agglomeration
            # constant kernel if kernel_function is none
            if kernel_function is None:
                for sample in range(AggNo):
                    # check if enough particles are available for agglomeration
                    if self.particles.shape[0] < 2:
                        break
                    # take 2 random samples and combine to single sample
                    samples = np.random.choice(self.particles, 2, replace=False)
                    self.particles = np.delete(self.particles, self.particles == samples[0])  # delete first sample
                    self.particles = np.delete(self.particles, self.particles == samples[1])  # delete second sample

                    if self.coordinate == 'L':
                        self.particles = np.append(self.particles, (samples[0] ** 3 + samples[1] ** 3) ** (
                                1 / 3))  # add new agglomerated sample
                    elif self.coordinate == 'V':
                        self.particles = np.append(self.particles, (samples[0] + samples[1]))  # add new agglomerated sample
                    else:
                        raise ValueError('Invalid coordinate. Please choose from L or V.')
            else:
                # choose two particles at random and perform accept reject sampling

                # normalize value
                n_points = 100  # Resolution of the grid
                sizes = np.linspace(1e-10, np.max(self.particles), n_points)
                XX, YY = np.meshgrid(sizes, sizes)
                max_kernel = np.max(kernel_function(XX,YY))

                for sample in range(AggNo):
                    # check if enough particles are available for agglomeration
                    if self.particles.shape[0] < 2:
                        break

                    agglomeration_accepted = False
                    while_counter = 0
                    while not agglomeration_accepted:
                        # take 2 random samples and combine to single sample
                        samples = np.random.choice(self.particles, 2, replace=False)

                        # accept if below kernel
                        kernel_value = kernel_function(*samples)
                        prob = kernel_value / max_kernel
                        if np.random.uniform() < prob:
                            self.particles = np.delete(self.particles, self.particles == samples[0])
                            self.particles = np.delete(self.particles, self.particles == samples[1])
                            if self.coordinate == 'L':
                                self.particles = np.append(self.particles, (samples[0] ** 3 + samples[1] ** 3) ** (1 / 3))  # add new agglomerated sample
                            elif self.coordinate == 'V':
                                self.particles = np.append(self.particles, sum_of_samples)  # add new agglomerated sample
                            else:
                                raise ValueError('Invalid coordinate. Please choose from L or V.')
                            agglomeration_accepted = True

                        # throw error if while loop runs too long
                        while_counter += 1
                        if while_counter > 50000:
                            print('Accept reject sampling for agglomeration did not converge. 1000 added to each crystal')
                            # generate very bad results if agglomeration did not converge
                            self.particles = self.particles + 10000
                            break


            # if AggNo > 0:
            #     print('Agglomeration event')

            # nucleation
            NuclNo = self.B * dt * PopnNo / self.m_0 + self.cut_off_nuc
            self.cut_off_nuc = NuclNo - int(NuclNo)
            NuclNo = int(NuclNo)

            # perform nucleation
            for _ in range(NuclNo):
                # add new sample with length 0 to list
                self.particles = np.append(self.particles, np.random.uniform(0, 1e-10))

            particle_list.append(self.particles.reshape(1, -1))

        time_end = time.perf_counter()

        self.d10 = np.percentile(self.particles, 10)
        self.d50 = np.percentile(self.particles, 50)
        self.d90 = np.percentile(self.particles, 90)

        # update moments
        self.mu = np.array([np.sum(self.particles ** k) for k in range(6)])

        if not (silence):
            # print some information about the simulation
            phenomena = ''
            if self.G:
                phenomena += 'growth'
            if self.B:
                if phenomena:
                    phenomena += ', '
                phenomena += 'nucleation'
            if self.beta:
                if phenomena:
                    phenomena += ', '
                phenomena += 'agglomeration'
            if self.Dil:
                if phenomena:
                    phenomena += ', '
                phenomena += 'dilution'
            if phenomena:
                phenomena = phenomena.split(', ')
                phenomena = ', '.join(phenomena[:-1]) + ', and ' + phenomena[-1] + '.'
            else:
                phenomena = 'no phenomena.'

            print('-----------------------------------')
            print(f'Simulation successful. Duration: {time_end - time_start:.2f}s')
            print('-----------------------------------')
            print(f'Simulation details:')
            print(f'Simulation for {phenomena}')
            print(f'Inner coordinate: {self.coordinate}')
            print(f'Simulated time: {self.t} with time step: {self.dt}')
            print(f'Number of particles at start of simulation: {no_particles_start}')
            print(f'Number of particles at end of simulation: {self.particles.shape[0]}')
            print(f'10th percentile (d10): {self.d10}')
            print(f'50th percentile (d50): {self.d50}')
            print(f'90th percentile (d90): {self.d90}')
            print(f'Width of distribution: {self.d90 - self.d10}')
            print('-----------------------------------')

        if return_full_simulation:
            return particle_list

    def plot_distribution(self, normalized: bool = True, bins: int = 50,
                          func: collections.abc.Callable[[float], float] = None, plot_char_d: bool = True) -> None:
        '''
        Plot the current distribution of the particles.

        Possibility to plot a function on top of the distribution.

        Parameters:
        - normalized: normalize the histogram (optional)
        - bins: number of bins for histogram (optional)
        - func: function to plot over histogram (e.g. an intial distribution or analytic solution) (optional)
        - plot_char_d: plot characteristic diameters (d10, d50, d90) (optional)
        '''

        # plot function if provided
        if func:
            x = np.linspace(*self.domain, 1000)
            plt.plot(x, func(x))

        # plot histogram
        # compute histogram
        hist, bin_edges = np.histogram(self.particles, bins=bins,
                                       density=normalized)  # , range=[0,self.particles.max()])
        # compute bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # plot histogram
        # pad bin_edges and hist with zeros, only if no nucleation but growth
        # bin_edges = np.concatenate(([0],[0.99*bin_edges[0]],bin_edges))
        # hist = np.concatenate(([0],[0],hist))

        plt.plot(bin_edges[:-1], hist, label='Monte Carlo')
        plt.xlim([0, self.particles.max()])

        # plot characteristic diameters
        if plot_char_d:
            plt.axvline(self.d10, color='r', alpha=0.5, label='d10', ls='--', lw=0.5)
            plt.axvline(self.d50, color='k', alpha=0.5, label='d50', ls='--', lw=0.5)
            plt.axvline(self.d90, color='b', alpha=0.5, label='d90', ls='--', lw=0.5)

        plt.xlabel('x')
        plt.ylabel('n(x)')
        plt.legend()
        plt.show()
