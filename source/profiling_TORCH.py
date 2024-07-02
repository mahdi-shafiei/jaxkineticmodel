import cProfile
from models.pCA_model_p20.pCA_fluxes_simpler import *
from trainer import Trainer
from functions.symplectic_adjoint.torch_symplectic_adjoint import odeint_checkpoint,odeint_adjoint,odeint_symplectic_adjoint,odeint_onecheckpoint

data=pd.read_csv("data/pCA_timeseries/pCA_fermentation_data_200424.csv",index_col=0)
parameters=pd.read_csv("results/pca_fermentation/lhs/pca_run1_optim_param_lhs_0.csv",index_col=0)
# parameters.index=list(parameter_dict.keys())
parameters=parameters.to_dict()['0']

fluxes=create_fluxes(parameter_dict) #very fast
metabolites_names=['GLUC','SHK',"TYR",'PHE','CIN','PCA','CO2','BIOMASS']

online_data=pd.read_excel("data/pCA_timeseries/Coumaric acid fermentation data for Paul van Lent.xlsx",sheet_name=3,header=2)


glucose_feed=online_data['Rate feed C (carbon source) (g/h)'].values[:-1]
t=online_data['Age (h)'].values[:-1]
fluxes['vGluc_Feed']=glucose_rate_polynomial_approximation(t,glucose_feed,N=40)

indices=np.arange(0,len(metabolites_names))
metabolites=dict(zip(metabolites_names,indices))

model=pCAmodel(fluxes,metabolites) #very fast

initial_values=torch.Tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

time_points=np.linspace(0,95.27555555664003,300)
tensor_timepoints=torch.tensor(time_points,dtype=torch.float64,requires_grad=False)

# print(parameters)
for name, param in model.named_parameters():
    if param.requires_grad:
        param.data=torch.Tensor([parameters[name]])
#         print(name)
#         param.data=torch.tensor(parameters[name],dtype=torch.float64,requires_grad=True)


loss_function_metabolites=[1,2,5,6,7]
loss_function_metabolites=[6]
trainer=Trainer(model,data,loss_func_targets=loss_function_metabolites,
                max_iter=5,err_thresh=1e-3,lr=1e-4,scaling=True,rtol=1e-5,atol=1e-7)

def main():
    import cProfile
    import pstats


    with cProfile.Profile() as pr:
        # model.forward(0,initial_values)
        trainer.train()
        # predicted_c =odeint_symplectic_adjoint(func=model, y0=initial_values,method="adaptive_heun", t=tensor_timepoints,rtol=1e-3,atol=1e-6)
        
    stats=pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()
    stats.dump_stats(filename="needs_profiling.prof")


if __name__=="__main__":
    main()




