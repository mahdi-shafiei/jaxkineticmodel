

from source.kinetic_mechanisms import JaxKineticMechanisms as jm
from source.building_models import JaxKineticModelBuild as jkm
from source.load_sbml.sbml_load import *
from source.load_sbml.sbml_model import SBMLModel
import jax.numpy as jnp
import jax
import numpy as np
from source.utils import get_logger
logger = get_logger(__name__)
import diffrax 
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import scipy
import sklearn
import xgboost as xgb
from xgboost import XGBClassifier,XGBRegressor






class DesignBuildTestLearnCycle:
    """A class that represents a metabolic engineering process. The underlying process is a kinetic model (parameterized and with initial conditions). Can be used
    to simulate scenarios that might occur in true optimization processes
    Input:
    1.  A model: either build or SBML
    2. Parameters: defined in a global way. This will represent the state of optimization process 
    3. Initial conditions for the model
    4. Time evaluation scale of the process. 
    
    
    """
    def __init__(self,
                 model,
                 parameters:dict,
                 initial_conditions :jnp.array,
                 timespan:jnp.array,
                 target:list):
        self.species_names=model.species_names
        self.kinetic_model=jax.jit(model.get_kinetic_model())
        self.parameters=parameters
        self.initial_conditions=initial_conditions
        self.timespan=timespan
        self.cycle_status=0
        self.library_units=None #library defines the building blocks of actions when constructing ME scenarios
        self.designs_per_cycle={}
        self.reference_production_value=None
        self.target=target
        self.reference=None  #
    


    def DESIGN_establish_library_elements(self, parameter_target_names, parameter_perturbation_values):
        """
        The actions that can be taken when sampling scenarios during the Design-phase.
        From an experimental perspective, this can be viewed as the library design phase.
        
        Input:
        - parameter_target_names: names of the parameters that we wish to perturb
        - parameter_perturbation_values: the actual perturbation (promoter) values of the parameters.
        These are defined RELATIVE to the reference state.
        """
        # Check that all parameter_target_names are valid
        for pt in parameter_target_names:
            if pt not in self.parameters.keys():
                logger.error(f"Parameter target {pt} not in the model. Perhaps a spelling mistake?")
                return None  # Return None and do not overwrite self.library_units

        # If all parameters are valid, flatten the combinations
        flattened_combinations = [
            (name, value)
            for name, values in zip(parameter_target_names, parameter_perturbation_values)
            for value in values
        ]
        
        # Create a DataFrame for the elementary actions
        elementary_actions = pd.DataFrame(flattened_combinations, columns=['parameter_name', 'promoter_value'])

        self.library_units = elementary_actions
        self.parameter_target_names=parameter_target_names
        return elementary_actions
    
    def DESIGN_assign_probabilities(self,occurence_list=None):
        """This functions assigns a probability to each element in the action list. Can be viewed as changing concentrations in a library design"""
        rows,cols=np.shape(self.library_units)
        if occurence_list is not None:
            if len(occurence_list)==rows:
                self.library_units['probability']=np.array(occurence_list)/np.sum(occurence_list)
                return_message="manual probabilities"
                pass 
            else:
                return_message="None"
                logger.error(f"Length of list of occurences of promoters is not matching ")
        else:
            return_message="equal probabilities"
            self.library_units['probability']=np.ones(rows)/rows
        return return_message
    
    def DESIGN_generate_strains(self,elements,samples,replacement=False):
        """Sample designs given the elementary actions given
        Input: number of elements to choose from the library (typically 6), number of samples.
        Replacement means whether we allow duplicate genes in the designs."""
        strains=[]
        strain_promoters=[]
        for i in range(samples):
            perturbed_parameters=self.parameters.copy()
            sample=self.library_units.sample(n=elements,weights=self.library_units['probability'],replace=replacement)[['parameter_name','promoter_value']]
            strain={}

            for param, value in zip(sample['parameter_name'].values, sample['promoter_value']):
                if param in strain:
                    strain[param] += value  # Sum the values if the key exists
                else:
                    strain[param] = value   # Add the new key-value pair if it doesn't exist

            #overwrite reference parameters.
            strain_promoter={}
            for key,values in strain.items():
                perturbed_parameters[key]=perturbed_parameters[key]*strain[key]
                strain_promoter[key]=strain[key]

            strains.append(perturbed_parameters)
            strain_promoters.append(strain_promoter)


        self.designs_per_cycle[f"cycle_{self.cycle_status}_designs"]=strain_promoters
        return strains

    def BUILD_simulate_strains(self, strains_perturbed, plot=False):
        """Simulates perturbations with respect to the reference strain. Takes the mean value of the last 10 simulated steps. We then save this into the designs_per_cycle status  """
        
        # Simulate the reference strain
        ys_ref = self.kinetic_model(self.timespan, self.initial_conditions, self.parameters)
        ys_ref = pd.DataFrame(ys_ref, columns=self.species_names)
        ys_ref = ys_ref[self.target]

        ys_ref_value=np.mean(ys_ref.iloc[-10:],axis=0)

        if plot:
            fig, ax = plt.subplots(figsize=(4, 4))

        
        # Loop through the perturbed strains and simulate each one
        simulated_values={str(i):[] for i in self.target}
        for strain_p in strains_perturbed:
            ys = self.kinetic_model(self.timespan, self.initial_conditions, strain_p)
            ys = pd.DataFrame(ys, columns=self.species_names)
            ys = ys[self.target]
            ys=ys/ys_ref

            ys_final=np.mean(ys.iloc[-10:],axis=0)
            
            for targ in self.target:
                simulated_values[targ].append(ys_final[targ])



            if plot:
                ax.plot(self.timespan, ys, label=f'Strain {strain_p}')  # Add a label for each strain if desired
        if plot:
            ax.set_title('Perturbed Strains')
            ax.set_xlabel('Time')
            ax.set_ylabel(f"{self.target}/Ref")
            plt.show()
        
        self.reference_production_value=ys_ref_value


        return simulated_values
    
        ### We now have simulated values that are strains. We want to have synthetic data that can be used to learn features in the data of importance

        ### we need to have a few functions:
        # a function that formats the generated dataset given the reference parameter set as well as values (TEST)
        # a function that can add noise to the measurements (TEST add noise)

    def TEST_add_noise(self,values,percentage,type="homoschedastic"):
        """add noise to the training set, to see the effect of noise models on performance. Includes homoschedastic or heteroschedastic noise for a certain percentage.
        Other experiment specific noise models could be added as well. One then needs to model the noise w.r.t to its screening value"""

        noised_values={}
        if type=="homoschedastic":            
            #look back whether this is actually the right way to do it
            for targ in self.target:

                values_new=np.random.normal(values[targ],percentage)
                values_new[values_new<0]=0

                noised_values[targ]=values_new

        if type=="heteroschedastic":
            #We assume that the noise level is given by X_m=D*X_true +X_true, where D is the percentage of deviation. We now model this as a simple gaussian, dependent on percentage*Xtrue
            for targ in self.target:

                values_new=np.random.normal(values[targ],percentage*np.array(values[targ]))
                values_new[values_new<0]=0
                noised_values[targ]=values_new

        return noised_values


    def TEST_format_dataset(self,strain_designs,production_values,reference_parameters):
        """Function that given strain designs and a reference strain (parameter set) formats the datasets as a pandas df for further use in ML/BO or whatever,
        the index will be coded with a cycle status coding. The last column is the.
        """

        strain_names=[f"cycle{self.cycle_status}_strain{i}" for i in range(len(strain_designs))]

        train_X=pd.DataFrame(strain_designs,index=strain_names)/reference_parameters
        train_X=train_X[self.parameter_target_names]
        
        for targ in self.target:
            train_X[f"Y_{targ}"]=production_values[targ]#/self.reference_production_value[targ]

        # print(train_x[self.target)
        return train_X
    
    ### Now the learning  and recommendation phase
    ### We would like an elegant way to include ML methods from outside the function (e.g., sklearn, xgboost)
    ### Or should we actually make an additional structure on top of Design-Build-Test-Learn-Cycle? The sort of Automated Lab structure
        
        
    ### following extra function required

    ### LEARN_train_model()
    ## Should perform cross-val
    ## Should be able to be used modular, such that we can use any ML method we want

    ### Learn_recommend_new_designs(): we should perhaps remain this open still

    ## LEARN_replace_reference_strain
    ## This should update the previous reference strain to perform further library or DoE transformation
    ### Also updates the DBTL cycle states
    ###
    def LEARN_train_model(self,data,target,args,model_type="XGBoost",test_size=0.2,runs=10):
        """Trains a model for the target given the datapoints.
        Input: data , target (variable to predicted), testset size (default 80/20 split, # runs for the r2)"""

        #train_test_val
        if target not in self.target:
            logger.error(f"Predictive variable {target} not in dataset")

        
        if model_type=="XGBoost":
            r2=[]
            for i in range(runs):
                params,alternative_params=args
                X=data[self.parameter_target_names]
                Y=data['Y_'+target]
                train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(X,Y, test_size=test_size,shuffle=True) 

                dtrain=xgb.DMatrix(train_x,label=train_y.to_list())
                dtest=xgb.DMatrix(test_x,label=test_y.to_list())

                ml_model=xgb.train(dtrain=dtrain,
                                params=params,
                                num_boost_round=alternative_params['num_boost_round'],
                                evals=[(dtrain,"train"),(dtest,"validation")],
                                early_stopping_rounds=alternative_params['early_stopping_rounds'],
                                verbose_eval=False)
                preds=ml_model.predict(dtest)
                true_value=dtest.get_label()
                slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(np.array(preds),np.array(true_value))
                r2.append(r_value**2)
        self.ml_model=ml_model
        return ml_model,r2
    

    def LEARN_validate_model(self,
                             samples,
                             elements,
                             target,
                            model_type="XGBoost",
                            plotting=True):
        
        """Validate the model with new data, generated from the same library distribution as before."""

        validation_set=self.DESIGN_generate_strains(elements=elements,samples=samples,replacement=True)
        validation_values=self.BUILD_simulate_strains(validation_set,plot=False)

        validation_data=self.TEST_format_dataset(strain_designs=validation_set,
                                production_values=validation_values,
                                reference_parameters=self.parameters)
        if model_type=="XGBoost":
            validation_set=xgb.DMatrix(validation_data[self.parameter_target_names])
            y_predicted=self.ml_model.predict(validation_set)

            _, _, r_value, _, _ = scipy.stats.linregress(np.array(y_predicted),np.array(validation_values[target]))
            if plotting==True:
                fig,ax=plt.subplots(figsize=(3,3))
                ax.scatter(validation_values['t6p'],y_predicted)
                ax.plot([0, 1], [0, 1], transform=ax.transAxes,linestyle='--',c='black')
                ax.set_title("R2 validation set")
                ax.set_xlabel("True (simulated values)")
                ax.set_xlabel("Predicted values)")
                ax.text(0.05, 0.95, f"R² = {np.round(r_value**2, 3)}", transform=ax.transAxes,
                        ha='left', va='top', fontsize=10, color='black')
                plt.show()
        return r_value**2