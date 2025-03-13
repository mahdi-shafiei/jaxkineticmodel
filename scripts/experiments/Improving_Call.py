"""Aiming to make jaxkineticmodel.__call__ more jaxlike """


import diffrax
from jaxkineticmodel.kinetic_mechanisms import JaxKineticMechanisms as jm
from jaxkineticmodel.building_models import JaxKineticModelBuild as jkm
import jax.numpy as jnp
import jax
import equinox
import matplotlib.pyplot as plt
import pandas as pd

ReactionA = jkm.Reaction(
    name="ReactionA",
    species=['A', 'B'],
    stoichiometry=[-1, 1],
    compartments=['c', 'c'],
    mechanism=jm.Jax_MM_Irrev_Uni(substrate="A", vmax="A_Vmax", km_substrate="A_Km"),
)

#Add reactions v1 to v3
v1 = jkm.Reaction(
    name="v1",
    species=['m1', 'm2'],
    stoichiometry=[-1, 1],
    compartments=['c', 'c'],
    mechanism=jm.Jax_MM_Irrev_Uni(substrate="m1", vmax="A_Vmax", km_substrate="A_Km"),
)

v2 = jkm.Reaction(
    name="v2",
    species=['m2', 'm3'],
    stoichiometry=[-1, 1],
    compartments=['c', 'c'],
    mechanism=jm.Jax_MM_Irrev_Uni(substrate="m2", vmax="B_Vmax", km_substrate="B_Km"),
)

v3 = jkm.Reaction(
    name="v3",
    species=['m2', 'm4'],
    stoichiometry=[-1, 1],
    compartments=['c', 'c'],
    mechanism=jm.Jax_MM_Irrev_Uni(substrate="m2", vmax="C_Vmax", km_substrate="C_Km"),
)

reactions = [v1, v2, v3]
compartment_values = {'c': 1}

# initialize the kinetic model object, and then make it a simulation object through jkm.NeuralODE
kmodel = jkm.JaxKineticModelBuild(reactions, compartment_values)
kmodel_sim = jkm.NeuralODEBuild(kmodel)
print(kmodel.stoichiometric_matrix)

ts = jnp.linspace(0, 10, 1000)
y0 = jnp.array([2, 0, 0, 0])
params = dict(zip(kmodel.parameter_names, jnp.array([1, 1, 1, 1, 1.5, 1])))

#jit the kmodel object. This results in a slow initial solve, but a c-compiled solve
kmodel_sim = jax.jit(kmodel_sim)
ys = kmodel_sim(ts, y0, params)
ys = pd.DataFrame(ys, columns=kmodel.species_names)
plt.plot(ys)
