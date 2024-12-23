from jaxkineticmodel.load_sbml.sbml_load import *
from jaxkineticmodel.load_sbml.sbml_model import SBMLModel
import matplotlib.pyplot as plt

filepath = ("../scripts/test.xml")

# load model from file_path
model = SBMLModel(filepath)
S=model._get_stoichiometric_matrix()

JaxKmodel = model.get_kinetic_model()


ts = jnp.linspace(0,10,2000)

# parameters in sbml can be either global or local parameters.
# For gradient descent purposes we want all of them global.
params = get_global_parameters(model.model)
params = {**model.local_params, **params}


#simulate given the initial conditions defined in the sbml
print(model._get_initial_conditions())
print(params)


ys = JaxKmodel(ts=ts,
            y0=jnp.array([0,0,0]),
            params=params)
ys=pd.DataFrame(ys,columns=S.index)

fig,ax=plt.subplots(figsize=(4,4))
for met in S.index:
    plt.plot(ys[met],label=met)
plt.legend()
plt.show()