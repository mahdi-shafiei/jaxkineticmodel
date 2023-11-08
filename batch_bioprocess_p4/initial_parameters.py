## Generate some parameter sets for the Batch Bioprocess model

import numpy as np
import pandas as pd

size=10
param1=np.random.uniform(low=-0.6,high=-0.0,size=size)
param2=np.random.uniform(low=0.000,high=0.08,size=size)
param3=np.random.uniform(low=-2.0,high=-1.2,size=size)
param4=np.random.uniform(low=-0.1,high=0.100,size=size)

parameter_dicts=[]
names=["qsmax","Ks","a","ms"]
for i in range(size):
    values=[param1[i],param2[i],param3[i],param4[i]]
    parameter_dict=dict(zip(names,values))
    parameter_dicts.append(parameter_dict)



parameter_sets=pd.DataFrame(parameter_dicts)
parameter_sets.to_csv("Batch_Bioprocess_parametersets.csv")
