# from frozen_dipole_model.frozen_dipole_model import frequencies, default_physical_parameters
import frozen_dipole_model as fd

physical_parameters = fd.default_physical_parameters


print(physical_parameters)
ho_physical = 100e-6 # in meters
to = 0.1 # in radians




#normalize

for normalization in ['hI', 'a', 'acrit']:
    parameters = fd.get_parameters(physical_parameters, normalization=normalization)

    ho = ho_physical / parameters[normalization]

    print('typical length scale', parameters[normalization] * 1e6, ' (um)')
    print('ho', ho, ' (norm)')


    freq = fd.frequencies(ho, to, physical_parameters, set_y_phi_zero=True, normalization=normalization, verbose=False)

    print(normalization, freq)

