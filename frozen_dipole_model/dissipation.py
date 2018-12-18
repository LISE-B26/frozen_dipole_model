import numpy as np

from frozen_dipole_model.frozen_dipole_model import  default_physical_parameters


def vortex_damping(fluxquantum=None, nV):



    gamma = (2*np.pi)**3 * np.pi/2 * fluxquantum**2 * Ai * nV**4 / (m * eta* etaV* wd**2 * delta**6)

    return gamma


def magnetic_damping():
    gamma = None

    return gamma


def millibar_to_pascals(P_mbar):

    return P_mbar*100

def torr_to_pascals(P_torr):

    return P_torr*20265/152



def gas_damping(radius=None, Pgas=None, Tgas=None, mgas=None, kB=None, density=None):
    """

    :param radius:
    :param Pgas: pressure in Pascals
    :param Tgas:
    :param mgas:
    :param kB:
    :param density:
    :return:
    """

    if radius is None:
        radius = default_physical_parameters['radius']
    if Pgas is None:
        Pgas = default_physical_parameters['Pgas']
    if Tgas is None:
        Tgas = default_physical_parameters['Tgas']
    if mgas is None:
        mgas = default_physical_parameters['mgas']
    if kB is None:
        kB = default_physical_parameters['kB']
    if density is None:
        density = default_physical_parameters['density']

    gamma = 2 * np.pi * 0.236 * np.sqrt(mgas/(kB * Tgas)) * Pgas / (radius * density)

    return gamma
