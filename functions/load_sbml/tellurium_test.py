
import tellurium as te

x=te.loadSBMLModel("../../models/SBML_models/fail_to_simulate/BIOMD0000000244_url.xml")
x.simulate(0,100,100)
x.plot()
