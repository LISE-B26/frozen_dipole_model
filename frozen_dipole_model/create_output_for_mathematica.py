
import frozen_dipole_model

import numpy as np

position = [0.1, 0.2, 1, 0.4, 2]
initial_conditions = [0.1,0.8]

position_eq = frozen_dipole_model.eq_position(*initial_conditions)

force_initial = frozen_dipole_model.force(position, *initial_conditions)
force_initial_num = frozen_dipole_model.force_num(position, *initial_conditions)


result = list(position)+list(initial_conditions)+list(position_eq)+list(force_initial)+list(force_initial_num)

np.savetxt("../data/for_mathematica.csv", result, delimiter="\t")

print(position_eq)
k = frozen_dipole_model.stiffness_matrix_num(position_eq, *initial_conditions, analytic_function='force')

np.savetxt("../data/for_mathematica_stiffness.csv", k, delimiter=",")


print(k)
# random positions and initial conditions, calculate the potential
data = []
for i in range(20):
    x = list(np.random.rand(3))+list(np.random.rand(2)*np.pi)+list(np.random.rand(1)*0.5+0.5)+list(np.random.rand(1)*np.pi)

    position, initial_conditions = x[0:5], x[5:]

    U = frozen_dipole_model.potential(position, *initial_conditions)

    force_initial = frozen_dipole_model.force_num(position, *initial_conditions, order=4)

    data.append(x+[U]+list(force_initial))
np.savetxt("../data/for_mathematica_potential.csv", data, delimiter=",")


