import numpy as np

from frozen_dipole_model.frozen_dipole_model import  default_physical_parameters



default_physical_parameters_dissipation = {

    'london_penetration_depth': 100e-9,  # London penetration depth in meters
    'coherence_length': 10e-9,  # Coherence length in meters
    'film_thickness': 100e-9,  # thin film thickness in meters

}

fluxquantum = default_physical_parameters['fluxquantum']   # flux quantum  kg m^2 / s^2 A

mu0 = default_physical_parameters['vacuum_permeability']  # H/m = kg m / (s^2 A^2)
kB = default_physical_parameters['kB']  # Boltzmann constant kg m^2 / s^2 K,
hbar = default_physical_parameters['hbar']  # Reduced Planck's constant [kg m^2 / s]

def calc_kp(pearl_length, a, verbose=False):
    """

    PRL 111, 145304 (2013) Eq. 21

    assumes 2 pi Lambda >> a

    for YBCO kp ~ 4.5x10^3 N/m^2

    :param fluxquantum:
    :param pearl_length: Pearl length Lambda
    :param a: lattice constant of vortex lattice
    :return:
    """


    if verbose:
        # check assumtion
        if 2*np.pi * pearl_length < 3*a:
            print('WARNING: assumption 2 pi Lambda >> a is violated!!\n (Lambda={:e}, a={:e})'.format(pearl_length, a))

    kp = fluxquantum ** 2 / (2 * mu0 * pearl_length * a ** 2)

    return kp


def calc_kint(levitation_height, dm, a, magnetic_moment):
    """

    spring constant between atom / nanomagnet and single vortex

    PRL 111, 145304 (2013) Eq. 74

    :param zo:
    :param dm:
    :param fluxquantum:
    :param a: vortex_lattice_constant
    :param magnetic_moment:
    :return:
    """

    zo=levitation_height

    # if we get a vector we just take the length
    if len(np.shape([magnetic_moment])) ==2:
        M = np.linalg.norm(magnetic_moment)
    else:
        M = magnetic_moment

    Delta = 2*np.pi * (zo+dm)/a
    Bo = fluxquantum / a**2
    return (2*np.pi)**3 * 3 * M * Bo / (a**2 * Delta**4)

def calc_pearl_length(london_penetration_depth, thickness_SC):
    """
    calculates the effective penetration depth (Pearl length) Lambda


    :param london_penetration_depth:
    :param thickness_SC:
    :return:
    """
    pearl_length = london_penetration_depth ** 2 / thickness_SC  # Pearl length (effective penetration depth)
    return pearl_length


def calc_A(mu):
    """

    PRL 111, 145304 (2013) Eq. 74

    :param mu: dipole vector (array of length 3)

    :return:
    """

    Ax = np.array([45/32, 15/32, 15/8])@ (mu**2)
    Ay = np.array([15/32, 45/32, 15/8]) @ (mu ** 2)
    Az = np.array([45/8, 15/8, 15/4]) @ (mu ** 2)

    return np.array([Ax, Ay, Az])

def calc_nv(magnetic_moment, levitation_height):

    # if we get a vector we just take the length
    if len(np.shape([magnetic_moment])) ==2:
        M = np.linalg.norm(magnetic_moment)
    else:
        M = magnetic_moment

    nv = M * mu0 / (2* levitation_height**3 * np.pi * fluxquantum)
    return nv

def calc_magnetic_moment(Br, radius):
    """

    :param Br: surface induction [Teslas]
    :param radius: particle radius [Meters]
    :return:
    """
    V = 4 * np.pi / 3 * radius**3  # m^3
    magnetic_moment = Br * V / mu0
    return magnetic_moment


def calc_thermal_amplitude(mass, frequency, temperature, is_sine=True):
    """

    :param mass:
    :param frequency:
    :param temperature:
    :return:
    """
    w = 2*np.pi*frequency
    # if we assume a square wave the factor two assures that
    # the sqrt of the variance of the signal equals np.sqrt(kB * temperature / (mass * w ** 2))
    if is_sine:
        amplitude = np.sqrt(2*kB*temperature / (mass * w**2))
    else:
        # for a Gaussian distribution we define the the amplitude as np.sqrt(kB * temperature / (mass * w ** 2))
        amplitude = np.sqrt(kB * temperature / (mass * w ** 2))  #
    return amplitude

def calc_mass(density, radius):
    """

    :param density: mass density [kg/m^3]
    :param radius: particle radius [Meters]
    :return:
    """
    V = 4 * np.pi / 3 * radius**3  # m^3
    mass = density * V
    return mass



def vortex_damping_jiggle(Bo, magnetic_moment, mass, levitation_height, thickness_SC, london_penetration_depth,
                          eta_bulk, vortex_lattice_constant,
                          verbose=False):
    """

    calculates the damping due to an oscillating magnetic field with amplitude Bo, the pulls on the vorticies


    :param magnetic_moment: magnetic moment of particle
    :param mass: particle mass [kg]
    :param levitation_height: distance between particle and SC [m]
    :param Bo: magnetic field amplitude

    :param nV: vortex density nV = 1/a^2 [1/m^2]
    :param thickness_SC: thickness of superconductor [m]
    :param london_penetration_depth: London penetration depth [m]
    :param vortex_lattice_constant: lattice constant of vortex lattice [m]
    :param eta: vortex viscosity coefficient eta = eta_bulk * pearl_length  - for YBCO eta ~ 3.5 x 10^-8 Ns / m^2
    :param magnet_orientation: orientation of magnet
    :return:
    """

    a = vortex_lattice_constant  # just to keep notation short

    # magnet_orientation = magnet_orientation / np.linalg.norm(magnet_orientation)  # make sure that we have a unit vector
    # V = 4*np.pi/3 * radius**3  # um^3
    #
    # magnetic_moment = calc_magnetic_moment(Br, radius)* magnet_orientation
    #
    # mass = density * V

    A =calc_A(magnetic_moment)
    pearl_length = calc_pearl_length(london_penetration_depth, thickness_SC)  # Pearl length
    # print('sss', pearl_length)
    # print('sss', a)
    eta = eta_bulk * pearl_length

    nV = 1/a**2

    kp = calc_kp(pearl_length, a)
    wd = kp / eta #  100e9 # depinning frequency wd = kp / eta
    delta = 2*np.pi * (levitation_height+pearl_length)/a  # normalized distance between particle and vortex, which sits at distance pearl_length below the surface

    gamma = (2*np.pi)**3 * np.pi/2 * A * Bo**2 / (mass * eta* wd**2 *a**4 * delta**6)

    gamma *=2*np.pi

    if verbose:
        print('pearl_length (nm)', 1e9*pearl_length)
        print('kp', kp)
        print('eta', eta)
        print('magnetic_moment',magnetic_moment)
        print('fluxquantum', fluxquantum)
        print('A', A)
        print('wd', wd)
        print('nV', nV)
        print('mass', mass)
        print('delta', delta)
        print('a', a)

    return gamma



def vortex_damping_drag(frequency, levitation_height, magnetic_moment, mass,
                        eta_bulk, london_penetration_depth, thickness_SC, temperature, vortex_lattice_constant,
                        verbose=False):
    """
    calculate the damping due to the magnet pulling on the vortecies


    :param frequency: trapping frequency [Hz
    :param levitation_height:
    :param magnetic_moment:
    :param mass:
    :param eta_bulk:
    :param london_penetration_depth:
    :param thickness_SC:
    :param temperature:
    :param vortex_lattice_constant:
    :return:
    """
    wt = 2*np.pi*frequency
    a = vortex_lattice_constant  # just to keep notation short

    xo = calc_thermal_amplitude(mass, frequency=frequency, temperature=temperature)
    pearl_length = calc_pearl_length(london_penetration_depth, thickness_SC)  # Pearl length
    eta = eta_bulk * pearl_length

    kp = calc_kp(pearl_length, a)
    kint = calc_kint(levitation_height, pearl_length, a, magnetic_moment)


    # gamma = (2*np.pi)**3 * np.pi/2 * fluxquantum**2 * Ai * nV**4 / (m * eta* etaV* wd**2 * delta**6)
    gamma = 1 / (2*hbar) * kint**2 * xo**2 * eta* wt / ( (kp+kint)**2 +eta**2*wt**2)

    if verbose:
        print('kp', kp)
        print('kint', kint)
        print('pearl_length', pearl_length)
        print('xo', xo)

    return gamma

#
# def magnetic_damping():
#     gamma = None
#
#     return gamma


def millibar_to_pascals(P_mbar):

    return P_mbar*100

def torr_to_pascals(P_torr):

    return P_torr*20265/152






def gas_damping(radius=None, Pgas=None, Tgas=None, mgas=None, density=None):
    """

    :param radius: particle radius [Meters]
    :param Pgas: pressure  [Pascals]
    :param Tgas: temperatrue [Kelvins]
    :param mgas: mass of gas molecules [kg]

    :param density: particle mass density [kg/m^3]]
    :return:
    """

    # if radius is None:
    #     radius = default_physical_parameters['radius']
    # if Pgas is None:
    #     Pgas = default_physical_parameters['Pgas']
    # if Tgas is None:
    #     Tgas = default_physical_parameters['Tgas']
    # if mgas is None:
    #     mgas = default_physical_parameters['mgas']
    # if density is None:
    #     density = default_physical_parameters['density']

    # print(radius, kB)

    gamma = 2 * np.pi * 0.236 * np.sqrt(mgas/(kB * Tgas)) * Pgas / (radius * density)

    return gamma


def calc_Qm(density, Br, chi_m_img, levitation_height, radius, frequency):
    omega = 2*np.pi*frequency
    return  density * mu0 / (chi_m_img * Br**2) * (2 * levitation_height / radius)** 8 * omega ** 2 * radius ** 2


def calc_Qsc(density, Br, chi_sc_img, levitation_height, radius, frequency, thickness_SC):
    omega = 2*np.pi*frequency
    volume_mag = 4*np.pi/3 * radius**3
    volume_SC = np.pi * radius ** 2 * thickness_SC
    return  density * mu0 / (chi_sc_img * Br**2) * (levitation_height / radius)** 8 * (volume_mag/volume_SC) * omega ** 2 * radius ** 2


# def gamma_dissplacement(mass, Tgas, frequency, Sdd):
#     wo = 2*np.pi*frequency
#     return np.pi*mass / (kB*Tgas)*wo**4*Sdd

def temperature_dissplacement(mass, frequency, Sdd, Q):
    wo = 2*np.pi*frequency
    return np.pi*mass / (2*kB)*wo**3*Q*Sdd