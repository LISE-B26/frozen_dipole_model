import numpy as np

from frozen_dipole_model.frozen_dipole_model import  default_physical_parameters



def vortex_creation_minimal_distance(radius=None, london_penetration_depth=None, film_thickness=None,
                                     vacuum_permeability=None, fluxquantum=None,
                                     coherence_length=None, Br=None):

    """

    calculate the minimum distance between dipole and superconductor to create a vortex

    Eq. (57) from SI of O. Romero-Isart et al. PRL 111, 145304 (2012)

    :param radius:
    :param london_penetration_depth:
    :param film_thickness:
    :param vacuum_permeability:
    :param fluxquantum:
    :param coherence_length:
    :param Br:
    :return:
    """



    if london_penetration_depth is None:
        london_penetration_depth = default_physical_parameters['london_penetration_depth']
    if film_thickness is None:
        film_thickness = default_physical_parameters['film_thickness']
    if vacuum_permeability is None:
        vacuum_permeability = default_physical_parameters['vacuum_permeability']
    if fluxquantum is None:
        fluxquantum = default_physical_parameters['fluxquantum']
    if coherence_length is None:
        coherence_length = default_physical_parameters['coherence_length']
    if radius is None:
        radius = default_physical_parameters['radius']
    if Br is None:
        Br = default_physical_parameters['Br']

    volume = 4 * np.pi / 3 * radius ** 3

    L = london_penetration_depth**2 / film_thickness

    md = Br*volume/vacuum_permeability

    return L * np.sqrt(vacuum_permeability * md/(fluxquantum * L) / np.log(L/coherence_length))


