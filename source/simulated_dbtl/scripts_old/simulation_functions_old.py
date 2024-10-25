#load packages
from pytfa.io.json import load_json_model
from skimpy.io.yaml import  load_yaml_model
from skimpy.analysis.oracle.load_pytfa_solution import load_concentrations, load_fluxes
from skimpy.core.parameters import ParameterValues
from skimpy.utils.namespace import *
import pandas as pd
import numpy as np
import skimpy
import time
import matplotlib.pyplot as plt
import itertools
import matplotlib
import sys
import os

from skimpy.core.modifiers import *
from skimpy.io.yaml import load_yaml_model
from skimpy.core.reactor import Reactor
from skimpy.analysis.oracle.load_pytfa_solution import load_concentrations, load_fluxes
from skimpy.viz.plotting import timetrace_plot
from pytfa.io.json import load_json_model
from skimpy.io.yaml import load_yaml_model
from skimpy.analysis.oracle.load_pytfa_solution import load_concentrations
from skimpy.core.parameters import load_parameter_population
from skimpy.simulations.reactor import make_batch_reactor
from skimpy.core.solution import ODESolutionPopulation
from skimpy.utils.namespace import *
from skimpy.viz.escher import animate_fluxes, plot_fluxes
import copy
from skimpy.io.yaml import export_to_yaml
from skimpy.analysis.ode.utils import make_flux_fun

def setup_ode_system(kmodel,tmodel,ref_solution):    
    # Units of the parameters are muM and hr
    CONCENTRATION_SCALING = 1e6  
    TIME_SCALING = 1 # 1hr to 1min
    DENSITY = 1200 # g/L
    GDW_GWW_RATIO = 0.3 # Assumes 70% Water
    
    #load relevant input files
    kmodel =  load_yaml_model(kmodel)
    tmodel = load_json_model(tmodel)
    ref_solution=pd.read_csv(ref_solution,index_col=0).loc['strain_1',:]
    
    # concentration of the metabolites |
    ref_concentrations = load_concentrations(ref_solution, tmodel, kmodel,
                                         concentration_scaling=CONCENTRATION_SCALING)
    #To run dynamic simulations, the model needs to contain compiled ODE expressions
    # help(kmodel.prepare()): Model preparation for different analysis types. The preparation is done
    #before the compling step
    parameter_values = {p.symbol:p.value for p in kmodel.parameters.values()}
    parameter_values = ParameterValues(parameter_values, kmodel)
    kmodel.prepare()
    kmodel.compile_ode(sim_type=QSSA, ncpu=8)
    
    for k in kmodel.initial_conditions:
        kmodel.initial_conditions[k] = ref_concentrations[k]
    
    return kmodel, ref_concentrations,tmodel


def setup_ode_system_simple(kmodel,ref_solution):
    kmodel=load_yaml_model(kmodel)
    ref_solution=pd.read_csv(ref_solution,index_col=0)
    ref_solution=ref_solution['Strain_1']
    parameter_values = {p.symbol:p.value for p in kmodel.parameters.values()}
    parameter_values = ParameterValues(parameter_values, kmodel)

    kmodel.prepare()
    kmodel.compile_ode(sim_type=QSSA, ncpu=8)

    for k in kmodel.initial_conditions:
        kmodel.initial_conditions[k]=ref_solution[k]
    return kmodel, ref_solution


def setup_batch_reactor(filenames):
    #Sets up the symbolic batch reactor

    reactor=make_batch_reactor(filenames['batch_file'])
    reactor.compile_ode(add_dilution=False)# check later what dilution does, it says dilution of intracellular metabolites
    tmodel=load_json_model(filenames['tmodel'])
    bkmodel=load_yaml_model(filenames['batch_kmodel'])
    reference_solutions=pd.read_csv(filenames['ref_solution'],index_col=0)
    ref_concentrations= load_concentrations(reference_solutions.loc['strain_1'], tmodel, bkmodel,
                                                      concentration_scaling=reactor.concentration_scaling)
    reactor.initial_conditions['biomass_strain_1'] = 0.1e12 # Number of cells
    reactor.initialize(ref_concentrations, 'strain_1')
    reactor.initialize(ref_concentrations, 'strain_2')
    return reactor, reference_solutions


def ode_integration(kmodel):
    sol=kmodel.solve_ode(np.linspace(0,800,10000),solver_type="cvode") 
    return sol

def perturb_kmodel(kmodel, enz_dict_perturb,parameter_values):
    ## Perturb the model given the design


    #this function works for both the kinetic model and reactor object
    n=len(enz_dict_perturb.keys())
    #this is required because python passes mutable objects
    kmodel.parameters = parameter_values 
    
    #perturb model: if only 1 value, or more perturbations
    perturbed_kmodel=kmodel #not overwriting the wt_model 
    if n==1:
        enz_label=list(enz_dict_perturb.keys())[0]
        enz_level=list(enz_dict_perturb.values())[0]
        perturbed_kmodel.parameters[enz_label].value=perturbed_kmodel.parameters[enz_label].value*enz_level

    else:
        enz_label=list(enz_dict_perturb.keys())
        enz_level=list(enz_dict_perturb.values())
        for i,k in enumerate(enz_label):
            perturbed_kmodel.parameters[k].value=perturbed_kmodel.parameters[k].value*enz_level[i]
    return perturbed_kmodel
    
    

def scenario_simulation(kmodel, designs,cart,params):
    flux_wt=params['flux_wt']
    if flux_wt==None:
    	print("Please provide the wildtype flux simulation in the params dictionary")
    parameter_values=params['parameter_values_wt']
    if flux_wt==None:
    	print("Please provide the wildtype parameter values in the params dictionary")
    enz_names=params['enz_names']
    if enz_names==None:
    	print("Please provide the the enzyme to target for engineering in the params dictionary")
    target_product=params['engineering_target']
    if target_product==None:
    	print("Please provide the target flux in the params dictionary")
    flux_fun=params['flux_fun']
    if flux_fun==None:
    	print("Please provide a skimpy flux function in the params dictionary")
    

    # Perturbation integration
    rel_flux_list = [] # list to store relative flux changes
    vmax_list = [] # list to store Vmax values
    
    for i in designs:
        # Create a perturbed model
        pmodel = perturb_kmodel(kmodel, i, parameter_values)
        # Integrate the perturbed model
        sol = ode_integration(pmodel)
        
        # Get Vmax values and store them in a dictionary
        perturbed_values = {p.symbol:p.value for p in pmodel.parameters.values()}
        vmax = dict(zip(list(parameter_values.keys()), list(parameter_values.values())))
        vmax_index = list(vmax.keys())
        vmax = list(vmax.values())
        vmax_list.append(vmax)
        
        # If the integration is successful
        if sol.ode_solution.message == 'Successful function return.':
            # Calculate the relative metabolite changes
            
            # Calculate the relative flux changes
            pmodel_parameters = {p.symbol:p.value for p in kmodel.parameters.values()}
            pmodel_parameters = ParameterValues(parameter_values, pmodel)
            for j, concentrations in sol.concentrations.iterrows():  
                flux_mt = flux_fun(concentrations, parameters=pmodel_parameters)
            rel_flux = relative_flux_change(flux_wt, flux_mt)
            rel_flux_list.append(rel_flux)
        else:
            # If the integration is unsuccessful, append NaN values to the lists
            my_flux_list = np.zeros(len(kmodel.reactions))
            my_flux_list[:] = np.NaN
            rel_flux_list.append(my_flux_list)
    
    cart = [tuple(i) for i in cart]
    # Create pandas dataframe to store the relative flux changes
    rel_flux_change = pd.DataFrame(np.array(rel_flux_list), index=cart, columns=flux_wt.keys())
    rel_flux_change.columns = list(flux_wt.keys())
    
    # Create pandas dataframe to store the Vmax values
    vmax = pd.DataFrame(np.array(vmax_list))
    vmax.columns = vmax_index
    vmax.index = cart
    vmax[target_product] = rel_flux_change[target_product]
    
    # Create training set dataframe with enzyme names and target product values
    training_set = pd.DataFrame(np.array(list(vmax.index)), columns=enz_names)
    training_set[target_product] = list(vmax[target_product])
    
    return rel_flux_change, training_set

def save_simulations(comb_cart,log_file):
    # This function takes two arguments: 
    #   - comb_cart: a pandas DataFrame containing simulation data
    #   - log_file: a string representing the filename where the simulation data will be saved

    # Write the simulation data to the log_file
    comb_cart.to_csv(log_file,mode="a",index=False,header=not os.path.exists(log_file))
    remove_duplicates=pd.read_csv(log_file)
    
    
    if list(comb_cart.columns)!=list(remove_duplicates.columns):
        print("Engineered parameter values are not similar. \n Make new simulation log file\n Remove current set of simulation from log file before continuing")
        pass
    else:
        
        remove_duplicates=remove_duplicates.drop_duplicates()
        remove_duplicates.to_csv(log_file,index=False)
        print("unique simulations saved")
    return remove_duplicates

def check_if_unique(train_x):
    my_list=[]
    number_of_elements=0
    indices_to_remove=[]
    """Check whether designs are all unique in the training data"""
    for i in range(np.shape(train_x)[0]):
        my_list.append(tuple(train_x.iloc[i,:]))
    for i in range(len(my_list)):
        for j in range(len(my_list)):
            if i!=j:
                if my_list[i]==my_list[j]:
                    number_of_elements+=1
                    indices_to_remove.append(i)
    return indices_to_remove,number_of_elements



def generate_perturbation_scheme(params):
    """Generates the combinatorial space (cartesian product) of all possible promoter-combinations
    
    Params
    - enz names: the genes to perturb
    - perturb_range: different promoters to be used
    """
    enz_names=params['enz_names']
    perturbation_matrix=params['perturb_range']
    ndim=len(enz_names)
    dim_range_list=[]
    for i in range(ndim):
        dim_range_list.append(np.array(perturbation_matrix[i]))
    #Cartesian product
    cart=[]
    for element in itertools.product(*dim_range_list):
        cart.append(np.array(element))
    comb_designs=[]
    for i in range(len(cart)):
        values=tuple(cart[i])
        design=dict(zip(enz_names,values))
        comb_designs.append(design)
    comb_x=pd.DataFrame(cart,columns=enz_names)
    return comb_x,comb_designs
    

def relative_met_change(sol_wt,sol_ps):
    difference=sol_wt.species-sol_ps.species
    difference=np.mean(difference[-5:-1,:],0)
    wt_avg=np.mean(sol_wt.species[-5:-1,:],0)
    x=difference/wt_avg
    rel_decrease=(1-x)
    return rel_decrease

def relative_flux_change(flux_wt,flux_mt):
    flux_w=np.array(list(flux_wt.values()))
    flux_m=np.array(list(flux_mt.values()))
    difference=flux_w-flux_m
    x=difference/flux_w
    rel_decrease=(1-x)
    
    return rel_decrease

def plot_energy_landscape(rel_change,metabolite,enzymes,enz_ind):
    coordinate_list=[]
    #find the coordinates (number of enzymes), and put them in a list
    for i in range(np.shape(rel_change[metabolite])[0]):
        coordinates=rel_change[metabolite].index[i]
        coordinate_list.append(coordinates)
    number_of_dimensions=len(coordinate_list[0])
    unique_heatmap_dims=[0]*number_of_dimensions
    for i in range(number_of_dimensions):
        dim_list=[]
        for k in range(len(coordinate_list)):
            ith_dim_coord=coordinate_list[k][i]
            dim_list.append(ith_dim_coord)
        dim_list=np.unique(dim_list)
        unique_heatmap_dims[i]=dim_list
    #print(unique_heatmap_dims)     
    matrix=np.zeros((len(unique_heatmap_dims[0]),len(unique_heatmap_dims[0])))
    matrix=pd.DataFrame(matrix, index=list(unique_heatmap_dims[enz_ind[0]]) ,
                                           columns=list(unique_heatmap_dims[enz_ind[1]]))
    #now for the coordinates, find the coordinate of the metabolite of interest index i
    #fill in the matrix based on the x_coordinate
    for i in range(np.shape(rel_change[metabolite])[0]):
        coordinates=rel_change[metabolite].index[i]
        #x_coordinate: vmax_forward_pfk will become the rows
        #ycoordinate: vmax_forward_LDH_d will become the columns
        x_coord=np.where(coordinates[enz_ind[0]]==matrix.index)[0] 
        y_coord=np.where(coordinates[enz_ind[1]]==matrix.columns)[0]
        temp_mat=np.array(matrix)
        #fill in the values of the list 
        temp_mat[x_coord,y_coord]=rel_change[metabolite].values[i]
        matrix=pd.DataFrame(temp_mat, index=list(unique_heatmap_dims[enz_ind[0]]) ,
                                           columns=list(unique_heatmap_dims[enz_ind[1]])) 
    #Apparently, here it goes wrong
    fig,ax=plt.subplots()
    im = ax.imshow(matrix.T,cmap='Reds')
    fig.colorbar(im)
    ax.set_xticks(np.arange(np.shape(matrix)[0]))
    ax.set_xticklabels(np.array(matrix.index))
    ax.set_yticks(np.arange(np.shape(matrix)[0]))
    ax.set_yticklabels(np.array(matrix.columns))
    ax.invert_yaxis()
    plt.xlabel(enzymes[enz_ind[0]])
    plt.ylabel(enzymes[enz_ind[1]])
    plt.xticks(rotation=-45)
    plt.title("Relative flux change in "+metabolite)
    plt.legend()
    plt.show()
    return matrix,fig







