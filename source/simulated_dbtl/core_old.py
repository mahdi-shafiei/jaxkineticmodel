from pytfa.io.json import load_json_model
from skimpy.io.yaml import  load_yaml_model
from skimpy.analysis.oracle.load_pytfa_solution import load_concentrations, load_fluxes
from skimpy.core.parameters import ParameterValues
from skimpy.utils.namespace import *
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
import os
import pandas as pd
import numpy as np

#import seaborn as sns
import skimpy
import time
import matplotlib.pyplot as plt
import itertools
import matplotlib
import sys
sys.path.insert(1, 'functions/')

# benchmark functions
import simulation_functions as sf
import scenarios as sc
import visualizations as vis
import noise as noise
import comb_sampling as cs

#ML methods
from sklearn import svm
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import  AdaBoostRegressor
from scipy.stats import linregress

from skopt import BayesSearchCV
import seaborn as sns
# from scipy.stats import entropy

from scipy.integrate import simpson
# from scipy.stats import entropy
from scipy.special import entr





def add_training_data(training_df,params, save_file=None):
    """Adds training data in an iterative fashion.
    Params
    training_df: the already existing training set. If no training set has been generated yet, start with an empty training set
    params: parameters such as sampling strategy (equal or manual), noise model, noise percentage, engineering target (what you try to optimize)
    save_simulation: whether to save simulation in log file (nice for when you do a lot of simulations and you do not want to generate a separate test set every time)
    save_file: simulation log file name
    """
    
    params2=params
    engineering_target=params2['engineering_target']
    save_simulation=params2['save_simulation']
    kmodel=params2['kmodel']
    enz_names=params2['enz_names']
    # sample designs and simulate them
    if params['sampling_strategy']=="equal":
        cyc_designs,cyc_cart=sc.equal_sampling_scenario(params['enz_names'],params['perturb_range'],params['N_designs'])
        training_cyc,training_cart=sf.scenario_simulation(kmodel,cyc_designs,cyc_cart,params)
        if save_simulation==True:
            simulation_log_data=sf.save_simulations(training_cart,save_file)#save without noise for better method comparison
            save_file=save_file.replace(".csv","_config.txt")
            pd.Series(params).to_csv(save_file)

    elif params['sampling_strategy']=="manual":
        cyc_designs,cyc_cart=sc.manual_sampling_scenario(params['enz_names'],params['perturb_range'],params['sampling_distribution'],params['N_designs'])
        training_cyc,training_cart=sf.scenario_simulation(kmodel,cyc_designs,cyc_cart,params)
        if save_simulation==True:
            simulation_log_data=sf.save_simulations(training_cart,save_file) #save without noise for better method comparison
            save_file=save_file.replace(".csv","_config.txt")
            pd.Series(params).to_csv(save_file)
    
    elif params['sampling_strategy']=="continuous":
        print("continuous scenario")
        cyc_designs,cyc_cart=sc.uniform_continuous_scenario(params['enz_names'],params['perturb_range'],params['N_designs'])
        training_cyc,training_cart=sf.scenario_simulation(kmodel,cyc_designs,cyc_cart,params)
        if save_simulation==True:
            simulation_log_data=sf.save_simulations(training_cart,save_file) #save without noise for better method comparison
            save_file=save_file.replace(".csv","_config.txt")
            pd.Series(params).to_csv(save_file)


    #add noise
    if params['noise_model']=="homoschedastic":
        noise_G=noise.add_homoschedastic_noise(training_cyc[engineering_target],params['noise_percentage'])
    elif params['noise_model']=="heteroschedastic":
        noise_G=noise.add_heteroschedastic_noise(training_cyc[engineering_target],params['noise_percentage'])

    training_cart[params['engineering_target']]=noise_G
    dfs=[training_df,training_cart]
    training_df=pd.concat(dfs)

    #This checks whether the training df doesnt have similar designs included. Can throw an error sometimes, but I dont know why.
    remove_ind,number_of_elements=sf.check_if_unique(training_df)
    while number_of_elements!=0:
        print(number_of_elements)
        training_df=training_df.drop(training_df.index[remove_ind])
        params2['N_designs']=number_of_elements
        
        if params['sampling_strategy']=="manual":
            cyc_designs,cyc_cart=sc.manual_sampling_scenario(params2['perturb_range'],params2['N_designs'],params2['sampling_distribution'],params2['enz_names'])
        elif params['sampling_strategy']=="equal":
            cyc_designs,cyc_cart=sc.equal_sampling_scenario(params['enz_names'],params['perturb_range'],params['N_designs'])
        training_cyc,training_cart=sf.scenario_simulation(kmodel,cyc_designs,cyc_cart,params)
        
        if save_simulation==True:
            simulation_log_data=sf.save_simulations(training_cart,save_file) #save without noise for better method comparison
            save_file=save_file.replace(".csv","_config.txt")
            pd.Series(params).to_csv(save_file)
        #add noise
        if params2['noise_model']=="homoschedastic":
            noise_G=noise.add_homoschedastic_noise(training_cyc[engineering_target],params2['noise_percentage'])
        elif params2['noise_model']=="heteroschedastic":
            noise_G=noise.add_heteroschedastic_noise(training_cyc[engineering_target],params2['noise_percentage'])
        
        training_cart[params['engineering_target']]=noise_G
        dfs=[training_df,training_cart]
        training_df=pd.concat(dfs)
        remove_ind,number_of_elements=sf.check_if_unique(training_df)

    #training and test split
    train_x=training_df[enz_names]
    train_y=training_df[params['engineering_target']]
    return train_x, train_y,training_df

def train_model(params,train_x,train_y):
    if params['ML_model']=="GradientBoostingRegressor":
        #random forest 
        regr_gbr = BayesSearchCV(
        GradientBoostingRegressor(),
        {
            "min_samples_split":(2,3,4,5,6,7,8),
            "min_samples_leaf":(2,3,4,5,6,7,8),
            "max_depth": (1,2,3,4,5,7,8),
            "learning_rate":(0.00001,0.0001,0.001,0.01,0.1,0.2,0.3),
        },
        n_iter=40,
        cv=5)
        regr_gbr.fit(train_x,train_y)
    return regr_gbr

def test_performance(testset,model,params): #use the simulation log file for increasingly better performance estimates
    test_x=testset[params['enz_names']]
    test_y=testset[params['engineering_target']]
    prediction=model.predict(test_x)
    slope, intercept, r_value, p_value, std_err = linregress(test_y,prediction)
    score=r_value**2
    return score


def get_probability_distribution(a3d_freq_mat,model,params):
    """Generates a probability distribution from the predicted combinatorial space.

    Args:
        a3d_freq_mat: A matrix of frequencies.
        params: A dictionary of parameters.
        plotting: A boolean flag indicating whether to enable plotting (default is False).

    Returns:
        probability_matrix: A matrix representing the probability distribution.
        feature_importance: A dictionary mapping enzyme names to their corresponding feature importance (entropy).
    """
    comb_x,comb_designs=sf.generate_perturbation_scheme(params)
    comb_y=model.predict(comb_x)
    comb_x[params['engineering_target']]=comb_y
    pred_comb_space=comb_x
    matrix=a3d_freq_mat

    enzymes=np.arange(0,np.shape(matrix)[1])
    # Initialize an empty probability matrix
    probability_matrix=np.zeros((np.shape(matrix)[0],np.shape(matrix)[1]))
    enz_names=params['enz_names']

    # Iterate over each enzyme and promoter element
    for j in range(np.shape(matrix)[1]): #for each enzyme we need to see what the frequency matrix looks like at each threshold
        for i in range(np.shape(matrix)[0]): #check for each promoter element the frequency
            probability_list=[]
            # Check the frequency at each threshold
            for k in range(np.shape(matrix)[2]): 
                probability_list.append(matrix[i,enzymes[j],k])
            area_under_curve=simpson(probability_list, dx=0.005)/np.max(pred_comb_space[params['engineering_target']])
            probability_matrix[i,j]=area_under_curve
    
    # Mask NaN values in the probability matrix     
    probability_matrix=np.ma.array(probability_matrix,mask=np.isnan(probability_matrix))
    probability_matrix=probability_matrix/probability_matrix.sum(axis=0)
    
    equi_entropy_list=[]
    numb_of_promoters=np.sum((probability_matrix.data>0),0)
    for i in numb_of_promoters:
        # print(i)
        equi_prob=np.ones(i)
        equi_prob=equi_prob/i
        equi_entropy=entr(equi_prob).sum(axis=0)/np.log(2)
        equi_entropy_list.append(equi_entropy)
    #now calculate the entropy for each column (enzyme). Due to the fact that not every enzyme has a similar amount of promoters
    # we need to normalize this according to the maxent (uniform distribution) 
    # to make sure that the range is 0-1
    entropy_list=entr(probability_matrix).sum(axis=0)/np.log(2)
    entropy_list=1-(np.array(entropy_list)/np.array(equi_entropy_list))
    # print(entropy_list)
    # entropy_list=entropy_list/equi_entropy_list
    feature_importance=dict(zip(enz_names,entropy_list))
    probability_matrix=probability_matrix.data
    return probability_matrix,feature_importance


def format_sampling_distribution(probability_matrix):
    """A formatting function to make the params['sampling_distribution'] format"""
    sampling_distribution=[]
    for i in range(np.shape(probability_matrix)[1]):
        x=probability_matrix[:,i]
        x=x[~np.isnan(x)]
        sampling_distribution.append(x)
    sampling_distribution=np.array(sampling_distribution)
    return sampling_distribution


def DBTL(training_df,params,logfilename,plotting=False):
    """Wrapper function for the DBTL cycle simulation
    Input: training dataframe, parameters
    Output:
    training_df: the training dataframe for the next cycle
    enz_numbers: ordered enz numbers based on feature importance
    sorted_entropies: order entropies of the enz_numbers
    probability matrix: sampling distribution for the next cycle
    top100: intersection value of the top 100 prediction
    top100_designs: the designs in the predicted top 100 (to check when best design is found
    score: r2 value"""
    if params['save_simulation']==True:
    	print(True)
    	train_x,train_y,training_df=add_training_data(training_df, params,logfilename)
    else:
    	train_x,train_y,training_df=add_training_data(training_df, params)
    	
    model=train_model(params,train_x,train_y)
    R2_score=test_performance(logfilename,model,params)
    threshold_range,a3d_freq_mat,remaining_designs=cs.scan_combinatorial_space(model,params,0.0)
    probability_matrix,feature_importance=get_probability_distribution(a3d_freq_mat,model,params)
    #evaluate
    if plotting==True:
        vis.plot_probability_distribution(probability_matrix,params)
    probability_matrix=format_sampling_distribution(probability_matrix)
    #prepare next cycle
    # enz_numbers,sorted_entropies,probability_matrix,matrix_pred=get_probability_matrix(prediction,params,plotting=False)  
    return training_df,probability_matrix,R2_score,model


