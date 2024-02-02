
## getting lower and upper bound values for parameters is in practice challenging. In order to consistently compare between performances, we will set bounds based on the true value
# ub is 2x the true value
# lb is 0.5x the true value



from functions.load_sbml.load_sbml_model import *
import pandas as pd

name="BIOMD0000000507_url.xml"
model=load_sbml_model("models/SBML_models/"+name)

parameters,boundaries,compartments=get_model_parameters(model)


lb={}
ub={}
for i in parameters:
    if i==0:
        i=0.0001
        lb[i]=parameters[i]*0.5
        ub[i]=parameters[i]*2
    lb[i]=parameters[i]*0.5
    ub[i]=parameters[i]*2

bounds=pd.DataFrame({'lb':lb,'ub':ub})
print(bounds)
name=name.replace(".xml","")
bounds.to_csv("parameter_initializations/"+name+"_bounds.csv")
