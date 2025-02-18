"""This code runs an export function on the kinetic model
We need from model.get_kinetic_model() the Jaxkmodel.func before jit compiling """

import libsbml
import jax
import jax.numpy as jnp
from typing import Union
from jaxkineticmodel.load_sbml.sympy_converter import SympyConverter, LibSBMLConverter
from jaxkineticmodel.load_sbml.jax_kinetic_model import NeuralODE
from jaxkineticmodel.building_models.JaxKineticModelBuild import NeuralODEBuild
from jaxkineticmodel.utils import get_logger

jax.config.update("jax_enable_x64", True)

logger = get_logger(__name__)


## design choice: separate export class instead of integrated with the SBML document
## this seems the way to go because when we use self-made models the sbml export


class SBMLExporter():
    """class used to export SBML model from a NeuralODE.JaxKineticModel object"""

    def __init__(self,
                 model: Union[NeuralODE, NeuralODEBuild]):
        assert isinstance(model, (NeuralODE, NeuralODEBuild))

        self.kmodel = model
        self.sympy_converter = SympyConverter()
        self.libsbml_converter = LibSBMLConverter()

        # we need to add this to NeuralODE object
        self.compartment_values = {}  # Need to deal with compartments for species for both NeuralODE and NeuralODEBuild

    def export(self,
               initial_conditions: jnp.ndarray,
               parameters: dict):
        """Exports model based on the input arguments to .xml file
        Input:
        - initial_conditions: initial conditions of the model
        - parameters: global parameters of the model"""
        try:
            document = libsbml.SBMLDocument()
        except:
            raise SystemExit('Could not create SBML document')

        export_model = document.createModel()

        #initial conditions and the compartments species belong to
        # (non-constant, non-boundary species)
        initial_conditions = dict(zip(self.kmodel.species_names, initial_conditions))
        species_compartments = self.kmodel.func.species_compartments  #same for both
        species_reference = {}



        #compartments: we need to retrieve compartment dictionary without interacting with
        compartments = [float(i) for i in self.kmodel.func.compartment_values]
        compartments = list(zip(species_compartments.values(), compartments))
        compartments = dict(set(compartments))

        for (c_id, c_size) in compartments.items():
            # Create a compartment inside this model, and set the required
            # attributes for an SBML compartment in SBML Level 3.

            c1 = export_model.createCompartment()
            check(c1, 'create compartment')
            check(c1.setId(c_id), 'set compartment id')
            check(c1.setConstant(True), 'set compartment "constant"')
            check(c1.setSize(c_size), 'set compartment "size"')
            check(c1.setSpatialDimensions(3), 'set compartment dimensions')
            # check(c1.setUnits('litre'),

        # we should save species we have made in a dictionary for later reference in
        #reactions


        for (s_id, s_comp) in species_compartments.items():
            s1 = export_model.createSpecies()
            check(s1, 'create species')
            check(s1.setId(s_id), 'set species id')
            check(s1.setCompartment(s_comp), 'set species s1 compartment')
            check(s1.setConstant(False), 'set "constant" attribute on s1')
            check(s1.setInitialAmount(float(initial_conditions[s_id])), 'set initial amount for s1')
            check(s1.setSubstanceUnits('mole'), 'set substance units for s1')
            check(s1.setBoundaryCondition(False), 'set "boundaryCondition" on s1')
            check(s1.setHasOnlySubstanceUnits(False), 'set "hasOnlySubstanceUnits" on s1')
            species_reference[s_id] = s1

        # one we have made a species, we should save it
        # in a dictionary for later reference in the reactions

        if isinstance(self.kmodel, NeuralODE):
            #do something
            logger.info(f"Exporting Neural ODE model of instance {type(self.kmodel)}")
            # print(self.kmodel.)

            # do something slightly different
            logger.info(f"Exporting Neural ODE model of instance {type(self.kmodel)}")
            self.kmodel.boundary_conditions

        # check(model, 'create model')
        # check(model.setTimeUnits("second"), 'set model-wide time units')
        #
        # # compartments are required. If a model does not define a compartmnet, we make a dummy compartments
        # compartments=self.model.compartments
        # if not compartments:
        #     compartments={'dummy':1}
        #
        # for (c_id, c_size) in compartments.items():
        #     # Create a compartment inside this model, and set the required
        #     # attributes for an SBML compartment in SBML Level 3.
        #
        #     c1 = model.createCompartment()
        #     check(c1, 'create compartment')
        #     check(c1.setId(c_id), 'set compartment id')
        #     check(c1.setConstant(True), 'set compartment "constant"')
        #     check(c1.setSize(c_size), 'set compartment "size"')
        #     check(c1.setSpatialDimensions(3), 'set compartment dimensions')
        #
        # # species (that are not boundaries and not constant)
        # # initial conditions for species are needed as an input
        # initial_conditions=dict(zip(self.model.species_names,initial_conditions))
        # species_reference= {}
        #
        # for (s_id, s_comp) in initial_conditions.items():
        #     s1 = model.createSpecies()
        #     check(s1, 'create species')
        #     check(s1.setId(s_id), 'set species id')
        #     # check(s1.setCompartment(s_comp), 'set species s1 compartment')
        #     check(s1.setConstant(False), 'set "constant" attribute on s1')
        #     check(s1.setInitialAmount(float(initial_conditions[s_id])), 'set initial amount for s1')
        #     check(s1.setSubstanceUnits('mole'), 'set substance units for s1')
        #     check(s1.setBoundaryCondition(False), 'set "boundaryCondition" on s1')
        #     check(s1.setHasOnlySubstanceUnits(False), 'set "hasOnlySubstanceUnits" on s1')
        #     species_reference[s_id] = s1
        #
        # print(self.model.boundary_conditions)
        #
        #


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
