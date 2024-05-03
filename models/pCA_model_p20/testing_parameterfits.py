
from pCA_fluxes_simpler import *

# parameter_dict={"v1_vmax":0.7,"v1_Kgluc":1.479116916657,"v1_Knh3":0.69203919172287,'v2_vmax':0.575218156072072,'v2_Kgluc':0.217746704816818,
#                 'v4_vmax':0.381420791046559,'v4_Kgluc':0.608636277914047,'v4_Ko2':0.170711681246758,
#                 'v5_vmax':0.0975382798333673,'v5_Kshk':0.819661617279053,'v5_Knh3':0.16221284866333,
#                 'v6_vmax':0.49102205027281,'v6_Kshk':0.348940074443817,'v6_Knh3':0.464994549751282,
#                 'v7_vmax':0.859700592405966,'v7_Kphe':0.0493920408189297,'v8_vmax':0.4,
#                 'v8_Kcin':0.2,'v8_Ko2':0.263323545455933

# }

parameters=pd.read_csv("results/pca_fermentation/lhs/pca_run1_optim_param_lhs_0.csv",index_col=0)
# parameters.index=list(parameter_dict.keys())
parameters=parameters.to_dict()['0']



fluxes=create_fluxes(parameter_dict)



# metabolites_names=['GLUC','SHK',"TYR",'PHE','CIN','PCA','CO2','NH3','O2','BIOMASS']
metabolites_names=['GLUC','SHK',"TYR",'PHE','CIN','PCA','CO2','BIOMASS']

online_data=pd.read_excel("data/pCA_timeseries/Coumaric acid fermentation data for Paul van Lent.xlsx",sheet_name=3,header=2)

glucose_feed=online_data['Rate feed C (carbon source) (g/h)'].values[:-1]
t=online_data['Age (h)'].values[:-1]
fluxes['vGluc_Feed']=glucose_rate_polynomial_approximation(t,glucose_feed,N=40)





indices=np.arange(0,len(metabolites_names))
metabolites=dict(zip(metabolites_names,indices))
model=pCAmodel(fluxes,metabolites)


# print(parameters)
for name, param in model.named_parameters():
    if param.requires_grad:
        param.data=torch.Tensor([parameters[name]])
#         print(name)
#         param.data=torch.tensor(parameters[name],dtype=torch.float64,requires_grad=True)


initial_values=torch.Tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

time_points=np.linspace(0,95.27555555664003,300)

# time_points=np.linspace(0,1,300)
tensor_timepoints=torch.tensor(time_points,dtype=torch.float64,requires_grad=False)

predicted_c =odeint_symplectic_adjoint(func=model, y0=initial_values,method="rk4", t=tensor_timepoints,rtol=1e-3,atol=1e-6)
# predicted_c =odeint_adjoint(func=model, y0=initial_values,t=tensor_timepoints,method="cvode",rtol=1e-3,atol=1e-8)

predicted_c=predicted_c.detach().numpy()

predicted_c=pd.DataFrame(predicted_c,columns=metabolites_names)


#true data
true_data=pd.read_csv("data/pCA_timeseries/pCA_fermentation_data_200424.csv",index_col=0)
ts_true_data=list(true_data.columns)

ts_true_data=[float(i) for i in ts_true_data]
true_data=true_data.T


# plt.plot()
# plt.plot(time_points,predicted_c['BIOMASS'],label="Biomass predicted")
# plt.scatter(ts_true_data,true_data['BIOMASS'],label="Biomass true")
# plt.plot(time_points,predicted_c['GLUC'],label="Glucose predicted")
# plt.scatter(ts_true_data,true_data['GLUC'],label="Glucose true")
plt.plot(time_points,predicted_c['CO2'],label="CO2 predicted")
plt.scatter(ts_true_data,true_data['CO2'],label="CO2 true")
# plt.plot(time_points,predicted_c['PCA'],label="PCA predicted")
# plt.scatter(ts_true_data,true_data['PCA'],label="PCA true")
plt.legend()
plt.show()