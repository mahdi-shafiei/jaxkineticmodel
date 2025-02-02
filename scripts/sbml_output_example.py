"""
 This script runs the first bit of code from our "Building Models"
 documentation and uses that to output SBML (very much work in progress;
 only a single compartment for now).
 The code for SBML output was lifted from SBML documentation.
"""

import libsbml

from jaxkineticmodel.kinetic_mechanisms import JaxKineticMechanisms as jm
from jaxkineticmodel.building_models import JaxKineticModelBuild as jkm
from jaxkineticmodel.load_sbml.sympy_converter import SympyConverter

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sp

ReactionA = jkm.Reaction(
    name="ReactionA",
    species=['A', 'B'],
    stoichiometry=[-1, 1],
    compartments=['c', 'c'],
    mechanism=jm.Jax_MM_Irrev_Uni(substrate="A", vmax="A_Vmax", km_substrate="A_Km"),
)

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
print(kmodel.stoichiometric_matrix)

# define the time interval, and the initial conditions

ts = jnp.linspace(0, 10, 1000)
y0 = jnp.array([2, 0, 0])
params = dict(zip(kmodel.parameter_names, jnp.array([1, 1, 1, 1, 1.5, 1])))

# jit the kmodel object. This results in a slow initial solve, but a c-compiled solve
kmodel_sim = jax.jit(kmodel_sim)
ys = kmodel_sim(ts, y0, params)
ys = pd.DataFrame(ys, columns=kmodel.species_names)

fig, ax = plt.subplots(figsize=(4, 4))
# ax.plot(ts,ys['m1'],label="m1")
ax.plot(ts, ys['m2'], label="m2")
ax.plot(ts, ys['m3'], label="m3")
ax.plot(ts, ys['m4'], label="m4")
ax.set_xlabel("Time (in seconds)")
ax.set_ylabel("Concentration (in mM)")
ax.legend()


# plt.show()


# ** Code below here lifted from SBML documentation **

def check(value, message):
    """Check output from libSBML functions for errors.

   If 'value' is None, prints an error message constructed using 'message' and then raises SystemExit.
   If 'value' is an integer, it assumes it is a libSBML return status code.
   If 'value' is any other type, return it unchanged and don't do anything else.

   For the status code, if the value is LIBSBML_OPERATION_SUCCESS, returns without further action; if it is not,
   prints an error message constructed using 'message' along with text from libSBML explaining the meaning of the
   code, and raises SystemExit.
   """
    if value is None:
        raise SystemExit('LibSBML returned a null value trying to ' + message + '.')
    elif type(value) is int:
        if value == libsbml.LIBSBML_OPERATION_SUCCESS:
            return
        else:
            err_msg = 'Error encountered trying to ' + message + '.' \
                      + 'LibSBML returned error code ' + str(value) + ': "' \
                      + libsbml.OperationReturnValue_toString(value).strip() + '"'
            raise SystemExit(err_msg)
    else:
        return


# Create an empty SBMLDocument object.  It's a good idea to check for
# possible errors.  Even when the parameter values are hardwired like
# this, it is still possible for a failure to occur (e.g., if the
# operating system runs out of memory).

sympy_converter = SympyConverter()

try:
    document = libsbml.SBMLDocument(3, 1)
except ValueError:
    raise SystemExit('Could not create SBMLDocument object')

# species = set()
# for r in reactions:
#    species |= set(r.species)


# Create the basic Model object inside the SBMLDocument object.  To
# produce a model with complete units for the reaction rates, we need
# to set the 'timeUnits' and 'extentUnits' attributes on Model.  We
# set 'substanceUnits' too, for good measure, though it's not strictly
# necessary here because we also set the units for individual species
# in their definitions.

model = document.createModel()
check(model, 'create model')
check(model.setTimeUnits("second"), 'set model-wide time units')
# check(model.setExtentUnits("mole"),       'set model units of extent')
# check(model.setSubstanceUnits('mole'),    'set model substance units')
#
# # Create a unit definition we will need later.  Note that SBML Unit
# # objects must have all four attributes 'kind', 'exponent', 'scale'
# # and 'multiplier' defined.
#
# per_second = model.createUnitDefinition()
# check(per_second,                         'create unit definition')
# check(per_second.setId('per_second'),     'set unit definition id')
# unit = per_second.createUnit()
# check(unit,                               'create unit on per_second')
# check(unit.setKind(libsbml.UNIT_KIND_SECOND),     'set unit kind')
# check(unit.setExponent(-1),               'set unit exponent')
# check(unit.setScale(0),                   'set unit scale')
# check(unit.setMultiplier(1),              'set unit multiplier')

compartments = {'c': 1}
for (c_id, c_size) in compartments.items():
    # Create a compartment inside this model, and set the required
    # attributes for an SBML compartment in SBML Level 3.

    c1 = model.createCompartment()
    check(c1, 'create compartment')
    check(c1.setId(c_id), 'set compartment id')
    check(c1.setConstant(True), 'set compartment "constant"')
    check(c1.setSize(c_size), 'set compartment "size"')
    check(c1.setSpatialDimensions(3), 'set compartment dimensions')
    # check(c1.setUnits('litre'),               'set compartment size units')

# species (that are not boundaries and not constant)
species_y0 = dict(zip(kmodel.species_names, y0))
species = {specimen: kmodel.species_compartments[specimen] for specimen in species_y0.keys()}

species_reference_dict = {}  # one we have made a species, we should save it
# in a dictionary for later reference
# in the reactions

for (s_id, s_comp) in species.items():
    s1 = model.createSpecies()
    check(s1, 'create species')
    check(s1.setId(s_id), 'set species id')
    check(s1.setCompartment(s_comp), 'set species s1 compartment')
    check(s1.setConstant(False), 'set "constant" attribute on s1')
    check(s1.setInitialAmount(float(species_y0[s_id])), 'set initial amount for s1')
    check(s1.setSubstanceUnits('mole'), 'set substance units for s1')
    check(s1.setBoundaryCondition(False), 'set "boundaryCondition" on s1')
    check(s1.setHasOnlySubstanceUnits(False), 'set "hasOnlySubstanceUnits" on s1')
    species_reference_dict[s_id] = s1

# boundary conditions
for (species_id, condition) in kmodel.boundary_conditions.items():
    compartment = kmodel.species_compartments[species_id]
    s1 = model.createSpecies()
    check(s1, 'create species')
    check(s1.setId(species_id), 'set species id')
    check(s1.setCompartment(compartment), 'set species s1 compartment')
    check(s1.setSubstanceUnits('mole'), 'set substance units for s1')

    # this is by definition of the boundary condition class True
    check(s1.setBoundaryCondition(True), 'set "boundaryCondition" on s1')
    check(s1.setHasOnlySubstanceUnits(False), 'set "hasOnlySubstanceUnits" on s1')

    if condition.is_constant:
        assert isinstance(condition.sympified, sp.Number)
        check(s1.setConstant(True), 'set "constant" attribute on s1')
        check(s1.setInitialAmount(float(condition.sympified)), 'set "initialAmount" attribute on s1')
    else:
        check(s1.setConstant(False), 'set "constant" attribute on s1')
        check(s1.setInitialAmount(jnp.nan), 'set "initialAmount" attribute on s1')

        math_ast = sympy_converter.sympy2libsbml(condition.sympified)
        rule = model.createAssignmentRule()
        check(rule.setVariable(s1.id), 'set "rule" attribute on s1')
        check(rule.setMath(math_ast), 'set "math" attribute on s1')
    species_reference_dict[species_id] = s1

# rule.setVariable("S1")  # The species ID to be governed by this rule
# rule.setMath(libsbml.parseL3Formula("time * 2"))  # Example formula: S1 = time * 2


# math_ast = libsbml.parseL3Formula('k * s1 * c1')
# check(math_ast,                           'create AST for rate expression')
#
# kinetic_law = r1.createKineticLaw()
# check(kinetic_law,                        'create kinetic law')
# check(kinetic_law.setMath(math_ast),      'set math on kinetic law')

parameter_names = kmodel.parameter_names

for parameter_name in parameter_names:
    p1 = model.createParameter()
    check(p1.setId(str(parameter_name)), 'set parameter name')
    check(p1.setConstant(True), 'set parameter name')
    check(p1.setValue(float(params[parameter_name])), 'set parameter value')

reactions = kmodel.reactions
for reaction in reactions:
    r1 = model.createReaction()
    check(r1, 'create reaction')
    check(r1.setId(str(reaction.name)), 'set reaction id')
    check(r1.setReversible(False), 'set reversible') #required
    check(r1.setFast(False), 'set reversible')
    for (s_id, stoich) in reaction.stoichiometry.items():
        if stoich < 0:
            species_ref1 = r1.createReactant()

        elif stoich > 0:
            species_ref1 = r1.createProduct()
        else:
            raise ValueError('stoich may not be 0')

        # use the dictionary with species references
        specimen = species_reference_dict[s_id]
        check(species_ref1, 'create reactant')
        check(species_ref1.setSpecies(specimen.getId()), 'set reactant species id')
        check(species_ref1.setConstant(specimen.getConstant()), 'set reactant species id')
        check(species_ref1.setStoichiometry(abs(stoich)), 'set absolute reactant/product stoichiometry')

        math_ast = sympy_converter.sympy2libsbml(reaction.mechanism.symbolic())

        kinetic_law = r1.createKineticLaw()
        check(kinetic_law,                        'create kinetic law')
        check(kinetic_law.setMath(math_ast),      'set math on kinetic law')

# # Create two species inside this model, set the required attributes
# # for each species in SBML Level 3 (which are the 'id', 'compartment',
# # 'constant', 'hasOnlySubstanceUnits', and 'boundaryCondition'
# # attributes), and initialize the amount of the species along with the
# # units of the amount.
#
# s1 = model.createSpecies()
# check(s1,                                 'create species s1')
# check(s1.setId('s1'),                     'set species s1 id')
# check(s1.setCompartment('c1'),            'set species s1 compartment')
# check(s1.setConstant(False),              'set "constant" attribute on s1')
# check(s1.setInitialAmount(5),             'set initial amount for s1')
# check(s1.setSubstanceUnits('mole'),       'set substance units for s1')
# check(s1.setBoundaryCondition(False),     'set "boundaryCondition" on s1')
# check(s1.setHasOnlySubstanceUnits(False), 'set "hasOnlySubstanceUnits" on s1')
#
# s2 = model.createSpecies()
# check(s2,                                 'create species s2')
# check(s2.setId('s2'),                     'set species s2 id')
# check(s2.setCompartment('c1'),            'set species s2 compartment')
# check(s2.setConstant(False),              'set "constant" attribute on s2')
# check(s2.setInitialAmount(0),             'set initial amount for s2')
# check(s2.setSubstanceUnits('mole'),       'set substance units for s2')
# check(s2.setBoundaryCondition(False),     'set "boundaryCondition" on s2')
# check(s2.setHasOnlySubstanceUnits(False), 'set "hasOnlySubstanceUnits" on s2')
#
# # Create a parameter object inside this model, set the required
# # attributes 'id' and 'constant' for a parameter in SBML Level 3, and
# # initialize the parameter with a value along with its units.
#
# k = model.createParameter()
# check(k,                                  'create parameter k')
# check(k.setId('k'),                       'set parameter k id')
# check(k.setConstant(True),                'set parameter k "constant"')
# check(k.setValue(1),                      'set parameter k value')
# check(k.setUnits('per_second'),           'set parameter k units')
#
# # Create a reaction inside this model, set the reactants and products,
# # and set the reaction rate expression (the SBML "kinetic law").  We
# # set the minimum required attributes for all of these objects.  The
# # units of the reaction rate are determined from the 'timeUnits' and
# # 'extentUnits' attributes on the Model object.
#
# r1 = model.createReaction()
# check(r1,                                 'create reaction')
# check(r1.setId('r1'),                     'set reaction id')
# check(r1.setReversible(False),            'set reaction reversibility flag')
# check(r1.setFast(False),                  'set reaction "fast" attribute')
#
# species_ref1 = r1.createReactant()
# check(species_ref1,                       'create reactant')
# check(species_ref1.setSpecies('s1'),      'assign reactant species')
# check(species_ref1.setConstant(True),     'set "constant" on species ref 1')
#
# species_ref2 = r1.createProduct()
# check(species_ref2,                       'create product')
# check(species_ref2.setSpecies('s2'),      'assign product species')
# check(species_ref2.setConstant(True),     'set "constant" on species ref 2')
#
# math_ast = libsbml.parseL3Formula('k * s1 * c1')
# check(math_ast,                           'create AST for rate expression')
#
# kinetic_law = r1.createKineticLaw()
# check(kinetic_law,                        'create kinetic law')
# check(kinetic_law.setMath(math_ast),      'set math on kinetic law')

# And we're done creating the basic model.
# Now return a text string containing the model in XML format.

# print(libsbml.writeSBMLToString(document))
sbml=libsbml.writeSBMLToString(document)
f = open("test.xml", "w")
f.write(sbml)
f.close()
