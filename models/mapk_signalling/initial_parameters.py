## Generate some parameter sets for the Batch Bioprocess model

import numpy as np
import pandas as pd

size=4


param1=np.random.uniform(low=49.99,high=50.01,size=size)
param2=np.random.uniform(low=49.99,high=50.01,size=size)
param3=np.random.uniform(low=0.299,high=0.310,size=size)
param4=np.random.uniform(low=0.2499,high=0.2501,size=size)
	
param5=np.random.uniform(low=7.99,high=8.01,size=size)
param6=np.random.uniform(low=0.024,high=0.026,size=size)
param7=np.random.uniform(low=14.99,high=15.01,size=size)
param8=np.random.uniform(low=0.749,high=0.751,size=size)

param9=np.random.uniform(low=14.99,high=15.01,size=size)
param10=np.random.uniform(low=0.029,high=0.031,size=size)
param11=np.random.uniform(low=14.99,high=15.01,size=size)
param12=np.random.uniform(low=0.499,high=0.501,size=size)

param13=np.random.uniform(low=14.99,high=15.01,size=size)



parameter_dicts=[]
names=["p_receptor_Vmax","p_receptor_Km","p_receptor_Ki","p_phosphatase1_Vmax",
"p_phosphatase1_Km","p_MAPKKKP_kcat","p_MAPKKKP_Km","p_phosphatase2_Vmax",
"p_phosphatase2_Km","p_MAPKK_Kcat","p_MAPKK_Km","p_phosphatase3_Vmax",
"p_phosphatase3_Km"]

for i in range(size):
    values=[param1[i],param2[i],param3[i],param4[i],param5[i],param6[i],param7[i],param8[i],param9[i],param10[i],param11[i],param12[i],param13[i]]
    
    parameter_dict=dict(zip(names,values))
    parameter_dicts.append(parameter_dict)



parameter_sets=pd.DataFrame(parameter_dicts)
parameter_sets.to_csv("mapk_parametersets.csv")
