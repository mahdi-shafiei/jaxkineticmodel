import diffrax
import numpy as np
import jax.numpy as jnp
import jax
from jaxkineticmodel.load_sbml.sbml_load import construct_param_point_dictionary, separate_params
from jaxkineticmodel.load_sbml.sbml_load import time_dependency_symbols


jax.config.update("jax_enable_x64", True)


class JaxKineticModel:
    def __init__(
        self,
        fluxes,
        stoichiometric_matrix,
        flux_point_dict,
        species_names,
        reaction_names,
        compartment_values,
    ):  # params,
        """Initialize given the following arguments:
        v: the flux functions given as lambidified jax functions,
        S: a stoichiometric matrix. For now only support dense matrices, but later perhaps add for sparse
        params: kinetic parameters
        flux_point_dict: a dictionary for each vi that tells what the
          corresponding metabolites should be in y. Should be matched to S.
        ##A pointer dictionary?
        """
        self.func = fluxes
        self.stoichiometry = stoichiometric_matrix
        # self.params=params
        self.flux_point_dict = flux_point_dict
        self.species_names = np.array(species_names)
        self.reaction_names = np.array(reaction_names)
        self.compartment_values = jnp.array(compartment_values)

    def __call__(self, t, y, args):
        """I explicitly add params to call for gradient calculations. Find out whether this is actually necessary"""
        params, local_params, time_dict = args

        # evaluate the time dictionary values at time t (for event functions e.g.)
        time_dict = time_dict(t)

        # function evaluates the flux vi given y, parameter, local parameters, time dictionary
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
        v = jnp.stack(
            [apply_func(i, y, self.flux_point_dict[i], local_params[i], time_dict[i]) for i in self.reaction_names]
        )  # perhaps there is a way to vectorize this in a better way
        dY = jnp.matmul(self.stoichiometry, v)  # dMdt=S*v(t)
        dY /= self.compartment_values
        return dY


class NeuralODE:
    func: JaxKineticModel

    def __init__(
        self,
            fluxes,
            stoichiometric_matrix,
            met_point_dict,
            v_symbols,
            compartment_values,
    ):
        self.func = JaxKineticModel(fluxes, jnp.array(stoichiometric_matrix), met_point_dict,
                                    list(stoichiometric_matrix.index), list(stoichiometric_matrix.columns),
                                    compartment_values)
        self.reaction_names = list(stoichiometric_matrix.columns)
        self.v_symbols = v_symbols
        self.Stoichiometry = stoichiometric_matrix
        self.max_steps = 300000
        self.rtol = 1e-7
        self.atol = 1e-10
        self.dt0=1e-11
        self.solver=diffrax.Kvaerno5()
        self.stepsize_controller=diffrax.PIDController(rtol=self.rtol, atol=self.atol,pcoeff=0.4, icoeff=0.3, dcoeff=0)

        def wrap_time_symbols(t):
            time_dependencies = time_dependency_symbols(v_symbols, t)
            return time_dependencies

        ## time dependencies: a function that return for all fluxes whether there is a time dependency
        self.time_dict = jax.jit(wrap_time_symbols)

    def _change_solver(self,solver,**kwargs):
        """To change the ODE solver object to any solver class from diffrax"""
        if isinstance(solver,diffrax.AbstractAdaptiveSolver):
            self.solver=solver
            step_size_control_parameters={'rtol':self.rtol, 'atol':self.atol,"pcoeff":0.4,"icoeff":0.3,"dcoeff":0}
            for key in kwargs:
                if key in step_size_control_parameters:
                    step_size_control_parameters[key] = kwargs[key]
            self.stepsize_controller=diffrax.PIDController(**step_size_control_parameters)
        return "solver added"



    def __call__(self, ts, y0, params):
        global_params, local_params = separate_params(params)

        # ensures that global params are loaded flux specific (necessary for jax)
        global_params = construct_param_point_dictionary(
            self.v_symbols, self.reaction_names, global_params
        )  # this is required,

        solution = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(self.func),
            solver=self.solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=self.dt0,
            y0=y0,
            args=(global_params, local_params, self.time_dict),
            stepsize_controller=self.stepsize_controller,
            saveat=diffrax.SaveAt(ts=ts),
            max_steps=self.max_steps,
        )

        return solution.ys
