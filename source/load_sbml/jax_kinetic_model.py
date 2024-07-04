import diffrax
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt
import numpy as np
import jax.numpy as jnp
import jax
from functools import partial
from .sbml_load import construct_param_point_dictionary, separate_params
from .sbml_load import time_dependency_symbols
from .sbml_load import *

jax.config.update("jax_enable_x64", True)



def create_fluxes_v(model):
    """This function defines the jax jitted equations that are used in TorchKinModel
    class
    """
    # retrieve whatever is important
    nreactions = model.getNumReactions()

    species_ic = get_initial_conditions(model)
    global_parameters = get_global_parameters(model)

    compartments = get_compartments(model)
    constant_boundaries = get_constant_boundary_species(model)

    lambda_functions = get_lambda_function_dictionary(model)
    assignments_rules = get_assignment_rules_dictionary(model)
    rate_rules=get_rate_rules_dictionary(model)
    event_rules=get_events_dictionary(model)


    v = {}
    v_symbol_dict = {}  # all symbols that are used in the equation.
    local_param_dict = {}  # local parameters with the reaction it belongs to as a new parameter

    for reaction in model.reactions:
        local_parameters = get_local_parameters(reaction)
        # print(local_parameters)
        # reaction_species = get_reaction_species(reaction)
        nested_dictionary_vi = {'species': species_ic,
                                'globals': global_parameters,
                                'locals': local_parameters,
                                'compartments': compartments,
                                'boundary': constant_boundaries,
                                'lambda_functions': lambda_functions,
                                'boundary_assignments': assignments_rules,
                                "rate_rules":rate_rules,
                                "event_rules":event_rules}  # add functionality

        vi_rate_law = get_string_expression(reaction)

        vi, filtered_dict = sympify_lambidify_and_jit_equation(vi_rate_law, nested_dictionary_vi)
        
        v[reaction.id] = vi  # the jitted equation
        v_symbol_dict[reaction.id] = filtered_dict

        # here
        for key in local_parameters.keys():
            newkey = "lp." + str(reaction.id) + "." + key
            local_param_dict[newkey] = local_parameters[key]
    return v, v_symbol_dict, local_param_dict





class JaxKineticModel:
    def __init__(self, v,
                 S,
                 flux_point_dict,
                 species_names,
                 reaction_names):  # params,
        """Initialize given the following arguments:
        v: the flux functions given as lambidified jax functions,
        S: a stoichiometric matrix. For now only support dense matrices, but later perhaps add for sparse
        params: kinetic parameters
        flux_point_dict: a dictionary for each vi that tells what the corresponding metabolites should be in y. Should be matched to S.
        ##A pointer dictionary?
        """
        self.func = v
        self.stoichiometry = S
        # self.params=params
        self.flux_point_dict = flux_point_dict  # this is ugly but wouldnt know how to do it another wa
        self.species_names = np.array(species_names)
        self.reaction_names = np.array(reaction_names)

    def __call__(self, t, y, args):
        """I explicitly add params to call for gradient calculations. Find out whether this is actually necessary"""
        params, local_params, time_dict = args
        time_dict = time_dict(t)

        # @partial(jax.jit, static_argnums=2)

        def apply_func(i, y, flux_point_dict, local_params, time_dict):
            if len(flux_point_dict) != 0:
                y = y[flux_point_dict]
                species = self.species_names[flux_point_dict]
                y = dict(zip(species, y))
            else:
                y = {}

            parameters = params[i]

            eval_dict = {**y, **parameters, **local_params, **time_dict}

            vi = self.func[i](**eval_dict)
            return vi

        # Vectorize the application of the functions
        indices = np.arange(len(self.func))

        # v = jax.vmap(lambda i:apply_func(i,y,self.flux_point_dict[i]))(indices)

        v = jnp.stack([apply_func(i, y, self.flux_point_dict[i],
                                  local_params[i],
                                  time_dict[i])
                       for i in self.reaction_names])  # perhaps there is a way to vectorize this in a better way
        dY = jnp.matmul(self.stoichiometry, v)  # dMdt=S*v(t)
        return dY


class NeuralODE:
    func: JaxKineticModel

    def __init__(self,
                 v,
                 S,
                 met_point_dict,
                 v_symbol_dictionaries):
        self.func = JaxKineticModel(v,
                                    jnp.array(S),
                                    met_point_dict,
                                    list(S.index),
                                    list(S.columns))
        self.reaction_names = list(S.columns)
        self.v_symbol_dictionaries = v_symbol_dictionaries
        self.Stoichiometry = S

        def wrap_time_symbols(t):
            time_dependencies = time_dependency_symbols(v_symbol_dictionaries, t)
            return time_dependencies

        self.time_dict = jax.jit(wrap_time_symbols)

    def __call__(self, ts, y0, params):
        global_params, local_params = separate_params(params)

        # ensures that global params are loaded flux specific (necessary for jax)
        global_params = construct_param_point_dictionary(self.v_symbol_dictionaries,
                                                         self.reaction_names,
                                                         global_params)  # this is required,

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Kvaerno5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            args=(global_params, local_params, self.time_dict),
            stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-12),
            saveat=diffrax.SaveAt(ts=ts),
            max_steps=50000
        )


        return solution.ys
