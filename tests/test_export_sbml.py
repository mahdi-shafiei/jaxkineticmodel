"""Test script for the export of sbml files and importing """

import jax.numpy as jnp
from jaxkineticmodel.load_sbml.sbml_model import SBMLModel
from jaxkineticmodel.kinetic_mechanisms import JaxKineticMechanisms as jm
from jaxkineticmodel.building_models import JaxKineticModelBuild as jkm
from jaxkineticmodel.load_sbml.export_sbml import SBMLExporter


def export_sbml1(filepath: str,model_name:str,
                 output_dir: str):

    """Test export of file directly from SBML model"""
    model = SBMLModel(filepath)
    # model.compile()
    JaxKmodel = model.get_kinetic_model()

    ###
    # Some manipulation, e.g. training/changing rate laws, etc..
    #then we want to export JaxKmodel.
    ###
    sbml = SBMLExporter(model=JaxKmodel)
    sbml.export(initial_conditions=model.y0,
                parameters=model.parameters,
                output_file=f"{output_dir}/{model_name}.xml")
    return True

def export_sbml2(output_dir: str,model_name: str):
    """Test export of file directly from self-build jaxkineticmodel"""
    # Add reactions v1 to v3
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
    kmodel = jkm.JaxKineticModel_Build(reactions, compartment_values)
    kmodel.add_boundary('m1', jkm.BoundaryCondition("0.5+0.3*sin(t)"))


    kmodel_sim = jkm.NeuralODEBuild(kmodel)
    y0 = jnp.array([2, 0, 0])
    params = dict(zip(kmodel.parameter_names, jnp.array([1, 1, 1, 1, 1.5, 1])))

    sbml = SBMLExporter(model=kmodel_sim)

    sbml.export(initial_conditions=y0,
                parameters=params,
                output_file=f"{output_dir}/{model_name}.xml")

    return True






def test_export_NeuralODE():
    model_name = "Smallbone2013_SerineBiosynthesis"
    filepath = f"models/sbml_models/working_models/{model_name}.xml"
    output_dir = "models/manual_implementations/export_sbml_test"

    assert export_sbml1(filepath, model_name, output_dir)

def test_export_sbml_NeuralODEBuild():
    model_name = "test.xml"
    output_dir = "models/manual_implementations/export_sbml_test"
    assert export_sbml2(output_dir, model_name)







#
# model_name="Smallbone2013_SerineBiosynthesis"
# filepath = (f"models/sbml_models/working_models/{model_name}.xml")
#
#
# # #%%
# from jaxkineticmodel.kinetic_mechanisms import JaxKineticMechanisms as jm
# from jaxkineticmodel.building_models import JaxKineticModelBuild as jkm
# from jaxkineticmodel.load_sbml.export_sbml import SBMLExporter
#
#
#
#
# ReactionA = jkm.Reaction(
#     name="ReactionA",
#     species=['A', 'B'],
#     stoichiometry=[-1, 1],
#     compartments=['c', 'c'],
#     mechanism=jm.Jax_MM_Irrev_Uni(substrate="A", vmax="A_Vmax", km_substrate="A_Km"),
# )
#
# # Add reactions v1 to v3
# v1 = jkm.Reaction(
#     name="v1",
#     species=['m1', 'm2'],
#     stoichiometry=[-1, 1],
#     compartments=['c', 'c'],
#     mechanism=jm.Jax_MM_Irrev_Uni(substrate="m1", vmax="A_Vmax", km_substrate="A_Km"),
# )
#
# v2 = jkm.Reaction(
#     name="v2",
#     species=['m2', 'm3'],
#     stoichiometry=[-1, 1],
#     compartments=['c', 'c'],
#     mechanism=jm.Jax_MM_Irrev_Uni(substrate="m2", vmax="B_Vmax", km_substrate="B_Km"),
# )
#
# v3 = jkm.Reaction(
#     name="v3",
#     species=['m2', 'm4'],
#     stoichiometry=[-1, 1],
#     compartments=['c', 'c'],
#     mechanism=jm.Jax_MM_Irrev_Uni(substrate="m2", vmax="C_Vmax", km_substrate="C_Km"),
# )
#
# reactions = [v1, v2, v3]
# compartment_values = {'c': 1}
#
# # initialize the kinetic model object, and then make it a simulation object through jkm.NeuralODE
# kmodel = jkm.JaxKineticModel_Build(reactions, compartment_values)
#
#
# kmodel.add_boundary('m1', jkm.BoundaryCondition("0.5+0.3*sin(t)"))
#
#
# kmodel_sim = jkm.NeuralODEBuild(kmodel)
# y0 = jnp.array([2, 0, 0])
# params = dict(zip(kmodel.parameter_names, jnp.array([1, 1, 1, 1, 1.5, 1])))
#
#
# model_name="test_model"
# sbml=SBMLExporter(model=kmodel_sim)
#
# sbml.export(initial_conditions=y0,
#             parameters=params,
#             output_file=f"{output_dir}/{model_name}.xml")
