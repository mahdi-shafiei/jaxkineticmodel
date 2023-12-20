## Generate some parameter sets for the Batch Bioprocess model

import numpy as np
import pandas as pd

#true={"v":-0.3,"p_Ks":0.01,"p_a":-1.6,"p_ms":-0.01}
size=400
param1=np.random.uniform(low=-0.5,high=-0.1,size=size)
param2=np.random.uniform(low=0.000,high=0.1,size=size)
param3=np.random.uniform(low=-1.8,high=-1.4,size=size)
param4=np.random.uniform(low=-0.2,high=0.0,size=size)






parameter_dicts=[]
names=["qsmax","Ks","a","ms"]
for i in range(size):
    values=[param1[i],param2[i],param3[i],param4[i]]
    parameter_dict=dict(zip(names,values))
    parameter_dicts.append(parameter_dict)


parameter_sets=pd.DataFrame(parameter_dicts)

parameter_sets.to_csv("Batch_Bioprocess_parametersets.csv")
