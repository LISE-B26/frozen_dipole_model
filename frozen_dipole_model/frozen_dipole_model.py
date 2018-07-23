from numpy import cos, sin, pi
from scipy.optimize import fmin
import numpy as np
from copy import deepcopy
from scipy.misc import derivative
import numdifftools as nd


default_dx = 1e-10

default_physical_parameters = {
    'Br' : 0.73,  # T
    'earth_acceleration' : 9.84,  #m/s^2
    'vacuum_permeability' : np.pi*4e-7,  # H/m
    'radius':22.5e-6,  # m
    'density':7600  # kg/m^3

}


def wrap(phases):
    """

    limit the phases to be in the interval -pi,pi

    :param phases:
    :return:
    """
    return (phases + np.pi) % (2 * np.pi) - np.pi

def to_physical_units(position, normalization, parameters):

    assert len(position)==3 or len(position)==5

    norm = parameters[normalization]

    if len(position)==3:
        return [p * norm for p in position[0:2]] + [position[2]]
    elif len(position)==5:
        return [p * norm for p in position[0:3]] + list(position[3:5])


def potential(position, ho, to, set_y_phi_zero=False, normalization = 'hI', alpha=None):
    """

    normalized potential multimply by Us to get actual potential

    Args:
        position: vector containing the position and orientation [x, y, z, theta, phi]
        ho: initial cooldown height
        to: initial oritentation theta_0
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
    elif normalization == 'acrit':
        U = alpha**(-3) * z


    denom_1 = cos(t)*((2*(ho+z)**2 -(x**2 + y**2))*cos(to) - 3*x*(ho + z)*sin(to))
    denom_2 = sin(t)*(3*(ho+z)*cos(to)*(x*cos(p)+y*sin(p))+sin(to)*(( (ho+z)**2 - 2*x**2+y**2)*cos(p) - 3*x*y*sin(p)) )

    U += (3+cos(2*t))/(6*z**3)-16*(denom_1 + denom_2) / (3*((z+ho)**2+x**2+y**2)**(5/2))

    assert isinstance(U, float)
    return U

def eq_position(ho, to, set_y_phi_zero=True, normalization= 'hI', alpha=None, verbose=False):
    """


    Args:
        ho: initial cooldown height
        to: initial oritentation theta_0
        set_x_phi_zero: from symmetry we know that y and phi should be zero, hence if True set y=phi=0
        initial position height ho and angle theta (xy and phi initial are 0 due to symmetry)

    Returns: the equilibrium position for a given initial conditions

    """
    inital_conditions = (ho, to, set_y_phi_zero, normalization, alpha)


    if set_y_phi_zero:
        #
        x0 = (0, ho, to)
    else:
        x0 = (0, 0, ho, to, 0)

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

def force(position, ho, to, alpha=1, normalization= 'hI'):
    """
    analytical calculation of the force (Jacobian)
    Args:
        position: vector containing the position and orientation [x, y, z, theta, phi]
        ho: initial cooldown height
        to: initial oritentation theta_0
        initial position height ho and angle theta (xy and phi initial are 0 due to symmetry)

    Returns: gradient along x, y, z, theta, phi

    """

    x, y, z, t, p = position  ## positions xyz and orientation theta and phi

    denom1 = 3*cos(t)*(x*(x**2 + y**2 - 4*(ho + z)**2)*cos(to) - (ho + z)*(-4*x**2 + y**2 + (ho + z)**2)*sin(to))
    denom2 = 3*sin(t)*((ho + z)*cos(to)*((-4*x**2 + y**2 + (ho + z)**2)*cos(p) - 5*x*y*sin(p))+ sin(to)*(x*(2*x**2 - 3*y**2 - 3*(ho + z)**2)*cos(p) - y*(-4*x**2 + y**2 + (ho + z)**2)*sin(p)))
    denom = 16 * (denom1 + denom2)
    fx = - denom / (3 * (x**2 + y**2 + (ho + z)**2)**(7/2))  # correct (checked with mathematica)

    denom1 =-y*cos(t)*((x**2 + y**2 - 4*(ho + z)**2)*cos(to) + 5*x*(ho + z)*sin(to))
    denom2 =   sin(t)*(y*cos(p)*(5*x*(ho + z)*cos(to) + (-4*x**2 + y**2 + (ho + z)**2)*sin(to)) - (x**2 - 4*y**2 + (ho + z)**2)*((ho + z)*cos(to) - x*sin(to))*sin(p))
    denom = 16* (denom1 + denom2)
    fy = denom / ((x**2 + y**2 + (ho + z)**2)**(7/2))  # correct (checked with mathematica)

    if normalization == 'hI':
        t1=1
    elif normalization == 'a':
        t1 = alpha
    elif normalization == 'acrit':
        t1 = alpha**(-3)

    t1 += - 3/(2*z**4) -  cos(2*t)/(2*z**4)
    denom1 = cos(t)*((ho + z)*(-3*(x**2 + y**2) + 2*(ho + z)**2)*cos(to) + x*(x**2 + y**2 - 4*(ho + z)**2)*sin(to))
    denom2 = sin(t)*(-(x**2 + y**2 -4*(ho + z)**2)*cos(to)*(x*cos(p) +y*sin(p)) + (ho +z)*sin(to)*((-4*x**2 + y**2 + (ho + z)**2)*cos(p) -5*x*y*(sin(p))))
    t2 = 16/(x**2 + y**2 + (ho + z)**2)**(7/2)*(denom1 + denom2)
    fz = t1 + t2  # correct (checked with mathematica)

    t1 = -(sin(2*t)/(3*z**3))
    denom1 = sin(t)*((x**2 + y**2 - 2*(ho + z)**2)*cos(to) + 3*x*(ho + z)*sin(to))
    denom2 = cos(t)*(3*(ho + z)*cos(to)*(x*cos(p) + y*sin(p)) + sin(to)*((-2*x**2 + y**2 + (ho + z)**2)*cos(p) - 3*x*y*sin(p)))
    t2 = -16/(3*(x**2 + y**2 + (ho + z)**2)**(5/2))*(denom1 + denom2)
    ft = t1 + t2  # correct (checked with mathematica)

    denom = -3*(ho + z)*cos(to)*(y*cos(p) -x*sin(p))
    denom += sin(to)*(3*x*y*cos(p) + (-2*x**2 + y**2 + (ho + z)**2*sin(p)))


    fp = 16*sin(t)*denom/(3*(x**2 + y**2 + (ho + z)**2)**(5/2))  # correct (checked with mathematica)

    return -np.array([fx, fy, fz, ft, fp])


def force_num(position, ho, to, order=2, verbose=False):
    force = -nd.Gradient(potential, order=order)(position, ho, to)
    return force


def get_parameters(physical_parameters, normalization= 'hI'):
    """

    Args:
        physical_parameters:

    Returns:

    """

    if not 'earth_acceleration' in physical_parameters:
        physical_parameters['earth_acceleration'] = 9.8  # m/s^2
    if not 'vacuum_permeability' in physical_parameters:
        physical_parameters['vacuum_permeability'] = np.pi*4e-7  # H/m

    for key in default_physical_parameters.keys():
        assert key in physical_parameters


    parameters = {'Us':None, 'mass_matrix':None, 'A':None}
    g = physical_parameters['earth_acceleration']
    Br = physical_parameters['Br']
    muo = physical_parameters['vacuum_permeability']
    a = physical_parameters['radius']
    density = physical_parameters['density']
    volume = 4*np.pi/3*a**3
    mass = volume * density

    moment_of_inertia = 2/5*mass*a**2

    hI = (a**3*Br**2/(16*g*density*muo))**(0.25)
    acrit = Br ** 2 / (16 * g * muo * density)



    parameters['mass_matrix'] = np.diag([mass, mass, mass, moment_of_inertia, moment_of_inertia])

    if normalization=='hI':
        parameters['A'] = np.diag([1. / hI, 1. / hI, 1. / hI, 1, 1])
        parameters['Us'] = mass * g * hI

    elif normalization=='a':
        parameters['A'] = np.diag([1. / a, 1. / a, 1. / a, 1, 1])
        parameters['Us'] = mass * g * acrit
    elif normalization == 'acrit':
        parameters['A'] = np.diag([1. / acrit, 1. / acrit, 1. / acrit, 1, 1])
        parameters['Us'] = mass * g * a**3 / acrit**(-2)


    parameters['hI'] = hI
    parameters['acrit'] = acrit
    parameters['moment_of_inertia'] = moment_of_inertia
    parameters['mass'] = mass
    parameters['alpha'] = a/acrit

    parameters['a'] = a

    return parameters

def stiffness_matrix_num(position, ho, to, analytic_function = 'force', normalization='hI', alpha=None):
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
        stiffness = nd.Hessian(potential)(position, ho, to, normalization=normalization, alpha=alpha)
    elif analytic_function == 'force':
        stiffness = -nd.Jacobian(force)(position, ho, to)  # negative sign because we define the force as the negative gradient
    elif analytic_function == 'k':
        print('NOT IMPLEMENTED YET')
    else:
        print('select one of k, potential or force')


    return stiffness


def frequencies(ho, to, physical_parameters, set_y_phi_zero=True, normalization='hI', verbose=False):
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

    position_eq = eq_position(ho, to, set_y_phi_zero=set_y_phi_zero, normalization=normalization, alpha=alpha)

    k = stiffness_matrix_num(position_eq, ho, to, normalization=normalization, alpha=alpha)






    Us = parameters['Us']

    print('position_eq (xyz in um)', to_physical_units(position_eq, normalization, parameters))
    print('===> Us', Us)

    A2 = parameters['A']**2

    inv_m = np.linalg.inv(parameters['mass_matrix'])

    # some consistency checks
    if verbose:
        print('checking consistency')
        assert np.sum(A2-np.dot(parameters['A'], parameters['A']))==0
        print(np.dot(inv_m, parameters['mass_matrix']))

        print(np.dot(A2 ,  inv_m) - A2 * inv_m)
        print('consistency check passed')

    W = Us * np.dot(A2 * inv_m, k)

    eigen_values = np.linalg.eigvals(W)

    return np.sqrt(eigen_values)/(2*np.pi)

