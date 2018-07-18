from numpy import cos, sin, pi
from scipy.optimize import fmin
import numpy as np
from copy import deepcopy
from scipy.misc import derivative
import numdifftools as nd


default_dx = 1e-10

default_physical_parameters = {
    'Br' : 0.73,  # T
    'earth_acceleration' : 9.8,  #m/s^2
    'vacuum_permeability' : np.pi*4e-7,  # H/m
    'radius':22.5e-6,  # m
    'density':7600  # kg/m^3

}

def potential(position, ho, to):
    """

    Args:
        position: vector containing the position and orientation [x, y, z, theta, phi]
        ho: initial cooldown height
        to: initial oritentation theta_0
        initial position height ho and angle theta (xy and phi initial are 0 due to symmetry)

    Returns:  the normalized potential for a given position and initial conditions

    """

    x, y, z, t, p = position ## positions xyz and orientation theta and phi


    denom_1 = cos(t)*((2*(ho+z)**2 -(x**2 + y**2))*cos(to) - 3*x*(ho + z)*sin(to))
    denom_2 = sin(t)*(3*(ho+z)*cos(to)*(x*cos(p)+y*sin(p))+sin(to)*(( (ho+z)**2 - 2*x**2+y**2)*cos(p) - 3*x*y*sin(p)) )

    # print('denom_', denom_1+denom_2)
    # print('x', -16*(denom_1 + denom_2) / (3*((z+ho)**2+x**2+y**2)**(5/2)))

    U = z + (3+cos(2*t))/(6*z**3)-16*(denom_1 + denom_2) / (3*((z+ho)**2+x**2+y**2)**(5/2))

    # print('denom_1', denom_1)
    # print('denom_2', denom_2)
    assert isinstance(U, float)
    return U

def eq_position(ho, to, verbose=False):
    """


    Args:
        ho: initial cooldown height
        to: initial oritentation theta_0
        initial position height ho and angle theta (xy and phi initial are 0 due to symmetry)

    Returns: the equilibrium position for a given initial conditions

    """

    x0 = (0, 0, ho, to, 0)

    inital_conditions = (ho, to)
#     res = minimize(type_II_superconductor_potential, x0, args=initial_consitions, jac=type_II_superconductor_force, tol=1e-6, bounds = bounds, options = {'disp':True})
    res = fmin(potential, x0, args=inital_conditions, xtol=1e-6, ftol=1e-17, disp=verbose, maxiter=1000)

    if verbose:
        print('fit result', res)
    
    return res


    
def force(position, ho, to):
    """
    analytical calculation of the force (Jacobian)
    Args:
        position: vector containing the position and orientation [x, y, z, theta, phi]
        ho: initial cooldown height
        to: initial oritentation theta_0
        initial position height ho and angle theta (xy and phi initial are 0 due to symmetry)

    Returns: gradient along x, y, z, theta, phi

    """
    
    x, y, z, t, p = position ## positions xyz and orientation theta and phi
    
    denom1 = 3*cos(t)*(x*(x**2 + y**2 - 4*(ho + z)**2)*cos(to) - (ho + z)*(-4*x**2 + y**2 + (ho + z)**2)*sin(to))
    denom2 = 3*sin(t)*((ho + z)*cos(to)*((-4*x**2 + y**2 + (ho + z)**2)*cos(p) - 5*x*y*sin(p))+ sin(to)*(x*(2*x**2 - 3*y**2 - 3*(ho + z)**2)*cos(p) - y*(-4*x**2 + y**2 + (ho + z)**2)*sin(p)))
    denom = 16 * (denom1 + denom2)
    fx = - denom / (3 * (x**2 + y**2 + (ho + z)**2)**(7/2))  # correct (checked with mathematica)

    denom1 =-y*cos(t)*((x**2 + y**2 - 4*(ho + z)**2)*cos(to) + 5*x*(ho + z)*sin(to))
    denom2 =   sin(t)*(y*cos(p)*(5*x*(ho + z)*cos(to) + (-4*x**2 + y**2 + (ho + z)**2)*sin(to)) - (x**2 - 4*y**2 + (ho + z)**2)*((ho + z)*cos(to) - x*sin(to))*sin(p))
    denom = 16* (denom1 + denom2)
    fy = denom / ((x**2 + y**2 + (ho + z)**2)**(7/2))  # correct (checked with mathematica)
    
    t1 = 1 - 3/(2*z**4) -  cos(2*t)/(2*z**4)
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


def force_num(position, ho, to, dx=default_dx):

    force = -nd.Gradient(potential)(position, ho, to)

    return force


def get_parameters(physical_parameters):
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


    parameters['mass_matrix'] = np.diag([mass, mass, mass, moment_of_inertia, moment_of_inertia])
    parameters['A'] = np.diag([1./hI, 1./hI, 1./hI, 1, 1])
    parameters['hI'] = hI
    parameters['moment_of_inertia'] = moment_of_inertia
    parameters['mass'] = mass
    parameters['Us'] = mass * g*hI
    return parameters

def stiffness_matrix_num(position, ho, to, dx=default_dx):
    """"""

    stiffness = nd.Hessian(potential)(position, ho, to)

    return stiffness

def frequencies(ho, to, physical_parameters, dx=default_dx, verbose=False):
    """


    Args:
        ho: initial cooldown height
        to: initial oritentation theta_0
        initial position height ho and angle theta (xy and phi initial are 0 due to symmetry)

    Returns: the equilibrium position for a given initial conditions

    """
    for key in physical_parameters.keys():
        assert key in physical_parameters

    position_eq = eq_position(ho, to)

    k = stiffness_matrix_num(position_eq, ho, to, dx=dx)

    parameters = get_parameters(physical_parameters)

    Us = parameters['Us']

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

    return W, k

