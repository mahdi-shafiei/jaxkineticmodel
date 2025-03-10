"""Functions that may be necessary for multiple classes etc.."""
import re
import collections

def construct_param_point_dictionary(v_symbol_dictionaries, reaction_names, parameters):
    """In jax, the values that are used need to be pointed directly in y."""
    flux_point_dict = {}
    for k, reaction in enumerate(reaction_names):
        v_dict = v_symbol_dictionaries[reaction]
        filtered_dict = {}
        for key, value in v_dict.items():
            if key in parameters.keys():
                filtered_dict[key] = parameters[key]

        flux_point_dict[reaction] = filtered_dict
    return flux_point_dict

def separate_params(params):
    """Seperates the global from local parameters using a identifier (lp.[Enz].)"""
    global_params = {}
    local_params = collections.defaultdict(dict)

    for key in params.keys():
        if re.match("lp_*_", key):
            fkey = key.removeprefix("lp_")
            list = fkey.split("_")
            value = params[key]
            newkey = list[1]
            local_params[list[0]][newkey] = value
        else:
            global_params[key] = params[key]
    return global_params, local_params


def separate_params_jac(params):
    """Only used to pass parameters locally and globally to the jacobian (see if this is better?)"""
    global_params = {}
    local_params = collections.defaultdict(dict)

    for key in params.keys():
        if re.match("lp.*.", key):
            # fkey = key.removeprefix("lp.")
            # list = fkey.split(".")
            value = params[key]
            # newkey = list[1]
            local_params[key] = value
        else:
            global_params[key] = params[key]
    return global_params, local_params

def get_global_parameters(model):
    """Most sbml models have their parameters defined globally,
    this function retrieves them"""
    params = model.getListOfParameters()
    global_parameter_dict = {param.id: param.value for param in params}
    return global_parameter_dict