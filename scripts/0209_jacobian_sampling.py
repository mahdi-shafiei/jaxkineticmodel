
import sys
sys.path.append("/home/plent/Documenten/Gitlab/NeuralODEs/jax_neural_odes")

from source.load_sbml.sbml_load import *
from source.load_sbml.sbml_model import SBMLModel
from source.parameter_estimation.jacobian import *
import jax
import matplotlib.pyplot as plt


filepath="models/sbml_models/working_models/Bertozzi2020.xml"
model=SBMLModel(filepath)
JaxKmodel = jax.jit(model.get_kinetic_model())

params = get_global_parameters(model.model)
params = {**model.local_params, **params}

    

jacobian_object=Jacobian(model)
compiled_jacobian=jacobian_object.compile_jacobian()
parameter_inits=pd.read_csv("parameter_initializations/initialization_succes/Bertozzi2020/Bertozzi2020_parameterset_id_lhs_N=1000run_2_log_update.csv",index_col=0)
filtered_parameters=jacobian_object.filter_parameter_sets(compiled_jacobian,parameter_inits,"stability")

print(np.shape(filtered_parameters))

ts=jnp.linspace(0,1000,10000)
y0=model.y0


ys=JaxKmodel(ts,y0,filtered_parameters.iloc[0,:].to_dict())

plt.plot(ts,ys)
plt.show()