from numpy import cos, sin, pi
from scipy.optimize import fmin
import numpy as np
from copy import deepcopy
from scipy.misc import derivative
import numdifftools as nd
from frozen_dipole_model.frozen_dipole_model import wrap, get_parameters


default_dx = 1e-10


def potential(position, dv, beta_v, set_y_phi_zero=False, normalization = 'hI', alpha=None):
    """

    normalized potential multimply by Us to get actual potential

    Args:
        position: vector containing the position and orientation [x, y, z, theta, phi]
        dv: depth of vortex
        beta_v: prefactor
        set_x_phi_zero: from symmetry we know that y and phi should be zero, hence if True set y=phi=0
        initial position height ho and angle theta (xy and phi initial are 0 due to symmetry)

        normalization = 'hI', 'a', 'acrit' select the normalization of the potential

    Returns:  the normalized potential for a given position and initial conditions

    """

    if set_y_phi_zero:
        ## positions xyz and orientation theta and phi
        x, z, t = position
        y, p = 0,0
    else:
        x, y, z, t, p = position ## positions xyz and orientation theta and phi


    if normalization == 'hI':
        U = z
    elif normalization == 'a':
        U = alpha * z
        print('warning: not tested for normalization a!!')
    elif normalization == 'acrit':
        U = alpha**(-3) * z
        print('warning: not tested for normalization acrit!!')

    U += (1/2+cos(2*t)/6)/(z**3)+beta_v*((dv+z)*cos(t) + (x*cos(p)+y*sin(p))*sin(t)) / (((z+dv)**2+x**2+y**2)**(3/2))

    assert isinstance(U, float)
    return U

def eq_position(dv, beta_v, set_y_phi_zero=True, normalization= 'hI', alpha=None, verbose=False):
    """


    Args:
        ho: initial cooldown height
        to: initial oritentation theta_0
        set_x_phi_zero: from symmetry we know that y and phi should be zero, hence if True set y=phi=0
        initial position height ho and angle theta (xy and phi initial are 0 due to symmetry)

    Returns: the equilibrium position for a given initial conditions

    """
    inital_conditions = (dv, beta_v, set_y_phi_zero, normalization, alpha)


    if set_y_phi_zero:
        #
        x0 = (0, 1, np.pi/2)
    else:
        x0 = (0, 0, 1, np.pi/2, 0)

#     res = minimize(type_II_superconductor_potential, x0, args=initial_consitions, jac=type_II_superconductor_force, tol=1e-6, bounds = bounds, options = {'disp':True})
    res = fmin(potential, x0, args=inital_conditions, xtol=1e-6, ftol=1e-17, disp=verbose, maxiter=1000)

    # limit phases to be in the intervall -pi, pi
    if set_y_phi_zero:
        res = res[0],res[1],wrap(res[2])
    else:
        res = res[0], res[1], res[2], wrap(res[3]), wrap(res[4])

    if verbose:
        print('fit result', res)


    return res


def force_num(position, order=2, verbose=False):
    force = -nd.Gradient(potential, order=order)(position)
    return force


def stiffness_matrix_num(position, analytic_function = 'force', normalization='hI', dv=None, beta_v=None, alpha=None):
    """

    :param position:
    :param ho:
    :param to:
    :return:
    """
    analytic_function = 'potential'

    if len(position) == 3:
        ## positions xyz and orientation theta and phi
        position = [position[0], 0, position[1], position[2], 0]

    if analytic_function == 'potential':
        stiffness = nd.Hessian(potential)(position, dv, beta_v, normalization=normalization, alpha=alpha)
    elif analytic_function == 'force':
        print('NOT IMPLEMENTED YET')
        # stiffness = -nd.Jacobian(force)(position,  dv, beta_v)  # negative sign because we define the force as the negative gradient
    elif analytic_function == 'k':
        print('NOT IMPLEMENTED YET')
    else:
        print('select one of k, potential or force')


    return stiffness


def frequencies(physical_parameters, set_y_phi_zero=True, normalization='hI', verbose=False):
    """


    Args:
        ho: initial cooldown height
        to: initial oritentation theta_0
        initial position height ho and angle theta (xy and phi initial are 0 due to symmetry)

    Returns: the equilibrium position for a given initial conditions

    """

    for key in physical_parameters.keys():
        assert key in physical_parameters

    parameters = get_parameters(physical_parameters, normalization=normalization)

    alpha = parameters['alpha']
    beta_v = parameters['beta_v']
    dv = parameters['flux_depth']

    position_eq = eq_position(dv=dv, beta_v=beta_v, set_y_phi_zero=set_y_phi_zero, normalization=normalization, alpha=alpha)

    k = stiffness_matrix_num(position_eq, normalization=normalization, dv=dv, beta_v=beta_v, alpha=alpha)


    # to stay within small numbers we multiply Us by 1e18 and inv_m by 1e-18, so the two factors compensate in the end
    Us = parameters['Us']*1e18
    inv_m = np.linalg.inv(parameters['mass_matrix'])*1e-18


    A2 = parameters['A'] ** 2

    W = np.dot(A2 * inv_m, k)  # all the values of this matrix should be close to 1, that's why we don't multiply by Us here

    ## we know that W is not symmetric so we cant use eigh
    eigen_values, eigen_vectors = np.linalg.eig(W)

    eigen_frequencies = np.sqrt(Us*eigen_values)/(2*np.pi)

    # order the freq by their main component in the order xyz theta phi
    freq_ordering = [np.argmax(abs(v)) for v in eigen_vectors]

    # check that all the each frequency has a main projection
    if np.allclose(sorted(freq_ordering), np.arange(5)):
        eigen_frequencies = eigen_frequencies[freq_ordering]
    else:
        print('warning, the frequencies do not have a unique projection keep unsorted ('+normalization+')', ho, to)


    return {
        'eigenfrequencies': eigen_frequencies,
        'position_eq':position_eq,
        'eigen_vectors':eigen_vectors
    }

    # if return_eq_positions:
    #     return eigen_frequencies, position_eq
    # else:
    #     return eigen_frequencies

