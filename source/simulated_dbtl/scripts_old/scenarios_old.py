
import regex as re
import pandas as pd
import numpy as np
import random
from collections import Counter
import pyDOE2


def equal_sampling_scenario(enz_names, perturb_range, N):
    """
    This function generates a list of designs for a scenario where each enzyme in a set of enzymes 
    has an equal chance of being perturbed within a certain range.
    
    :param enz_names: a list of strings representing enzyme names
    :param perturb_range: a list of tuples representing the perturbation range for each enzyme. 
    Each tuple contains two floats representing the minimum and maximum perturbation values.
    :param N: an integer representing the number of designs to generate
    
    :return: a list of dictionaries where each dictionary represents a design. Each dictionary has keys 
    that correspond to enzyme names and values that correspond to the perturbation value for that enzyme. 
    Additionally, the function returns a list of lists where each inner list represents a design and contains
    perturbation values for each enzyme.
    """
    library_choices=dict(zip(enz_names,perturb_range))# create a dictionary that maps each enzyme to its perturbation range
    cart=[] # create an empty list to hold the perturbation values for each design
    designs_list=[]
    for i in range(N):# loop N times to generate N designs
        design=[]
        for j in enz_names:
        
            x=np.random.choice(library_choices[j])# randomly select a perturbation value for the enzyme
            design.append(x)
        cart.append(design)
    for i in range(len(cart)):
        design=dict(zip(enz_names,cart[i]))# create a dictionary that maps enzyme names to perturbation values for the design
        designs_list.append(design)
    return designs_list,cart  # return the list of design dictionaries and the list of design perturbation values

def manual_sampling_scenario(enz_names,perturb_range,probability_distribution,N):

    """This function generates a list of designs for a scenario where the probability distribution 
    for perturbing each enzyme is manually specified.
    
    :param enz_names: a list of strings representing enzyme names
    :param perturb_range: a list of tuples representing the perturbation range for each enzyme. 
    Each tuple contains two floats representing the minimum and maximum perturbation values.
    :param prob_dist: a list of arrays representing the probability distribution for each enzyme. 
    Each array contains floats that sum up to 1 and represents the probability of perturbing each enzyme value.
    :param N: an integer representing the number of designs to generate
    
    :return: a list of dictionaries where each dictionary represents a design. Each dictionary has keys 
    that correspond to enzyme names and values that correspond to the perturbation value for that enzyme. 
    Additionally, the function returns a list of lists where each inner list represents a design and contains
    perturbation values for each enzyme. 
    """
    library_choices=dict(zip(enz_names,perturb_range))# create a dictionary that maps each enzyme to its perturbation range
    for i in range(len(probability_distribution)):
        if np.sum(probability_distribution[i])!=1:
            print("Distribution not adding up to 1, perform scaling for ",probability_distribution[i])
            new_probabilities=probability_distribution[i]/np.sum(probability_distribution[i]) #normalize based on magnitude of vector 
            probability_distribution[i]=new_probabilities

    cart=[] 
    designs_list=[]

    library_choices=dict(zip(enz_names,perturb_range))
    distribution_choices=dict(zip(enz_names,probability_distribution))
    for i in range(N):
        design=[]
        for j in enz_names:
            library_component=np.random.choice(library_choices[j],1,p=distribution_choices[j])[0]
            design.append(library_component)
        cart.append(design)

    for i in range(len(cart)):
        design=dict(zip(enz_names,cart[i]))# create a dictionary that maps enzyme names to perturbation values for the design
        designs_list.append(design)
        
    return designs_list,cart


    



### Visualization

colors = ['#00468BFF','#ED0000FF','#42B540FF','#0099B4FF','#925E9FFF','#FDAF91FF','#AD002AFF','#ADB6B6FF','#1B1919FF']
def plot_promoter_distribution(enz_names,cart):
    #change names for plotting
    enzymes=[]
    a_dict={}
    for i in enz_names:
        enz_names=i.replace("vmax_forward_","")
        enz_names=enz_names.replace("_"," ")
        enzymes.append(enz_names)
    cart=np.array(cart)
    for j in range(np.shape(cart)[1]):
        x=cart[:,j]
        counts=dict(Counter(x))
        a_dict[enzymes[j]]=counts
    #pd.DataFrame(a_dict).T.plot(kind="bar",stacked=True).legend(bbox_to_anchor=(1.0, 1.0)) 
    a_dict=pd.DataFrame(a_dict).sort_index()
    ax=a_dict.T.plot(kind="bar",stacked="True",color=colors)
    
    ax.legend(bbox_to_anchor=(1.0,1.0),title="Promoter Strength")
    ax.set_ylabel("Number of designs")

    return ax


def add_noise(fluxes,percentage):
    """Adds uniform noise to the observation proportional to the mean
    set flux to zero if it becomes negative"""
    error=np.random.uniform(low=fluxes*(1-percentage),high=fluxes*(1+percentage))-fluxes
    noised_fluxes=fluxes+error
    noised_fluxes[noised_fluxes<0]=0
    return noised_fluxes
    
def uniform_continuous_scenario(enz_names, perturb_range, N):
    """This function generates a list of designs for a scenario where each enzyme can be perturbed in a continuous (uniform) range between two values
    
    
    :param enz_names: a list of strings representing enzyme names
    :param perturb_range: a list of tuples representing the perturbation range for each enzyme. 
    Each tuple contains two floats representing the minimum and maximum perturbation values.
    :param N: an integer representing the number of designs to generate
    
    :return: a list of dictionaries where each dictionary represents a design. Each dictionary has keys 
    that correspond to enzyme names and values that correspond to the perturbation value for that enzyme. 
    Additionally, the function returns a list of lists where each inner list represents a design and contains
    perturbation values for each enzyme."""
    library_choices={}
    for i,enzyme in enumerate(enz_names):
        library_choices[enzyme]=[perturb_range[i][0],perturb_range[i][-1]]

    cart=[]
    designs_list=[]
    for i in range(N):# loop N times to generate N designs
        design=[]
        for enzyme in enz_names:
            x=np.random.uniform(low=library_choices[enzyme][0],high=library_choices[enzyme][1])
            design.append(x)
        cart.append(design)
    for i in range(len(cart)):
        design=dict(zip(enz_names,cart[i]))# create a dictionary that maps enzyme names to perturbation values for the design
        designs_list.append(design)
    return designs_list,cart  # return the list of design dictionaries and the list of design perturbation values


#### OLD functions

#def scenario2(perturb_range,N,enz_names):
#     #Choose with a certain probability distribution
#     cart=[]
#     designs_list=[]
#     for i in range(N):
#         x=np.random.choice(perturb_range,len(enz_names),p=[0.25,0.15,0.1,0.1,0.15,0.25])
#         x=tuple(x)
#         cart.append(x)
#     for i in range(len(cart)):
#         design=dict(zip(enz_names,cart[i]))
#         designs_list.append(design)
#     return designs_list,cart


# def scenario3(perturb_range,N,enz_names):
#     #Choose with a certain probability distribution
#     cart=[]
#     designs_list=[]
#     for i in range(N):
#         x=np.random.choice(perturb_range,len(enz_names),p=[0.1,0.15,0.25,0.25,0.15,0.1])
#         x=tuple(x)
#         cart.append(x)
#     for i in range(len(cart)):
#         design=dict(zip(enz_names,cart[i]))
#         designs_list.append(design)
#     return designs_list,cart

# def scenario4(levels,reduction):
#     #A fractional factorial approach
#     scenario4=pyDOE2.gsd(levels,reduction)

#     sc4=np.zeros((np.shape(scenario4)[0],np.shape(scenario4)[1]))
#     dictionary_for_sc4={0:1, 1:0.5 ,2:1.5 ,3:2}
#     for i in range(np.shape(scenario4)[0]):
#         for j in range(np.shape(scenario4)[1]):
#             sc4[i,j]=dictionary_for_sc4[scenario4[i,j]]
#     enz_names=["vmax_forward_Enzyme_A","vmax_forward_Enzyme_B","vmax_forward_Enzyme_C","vmax_forward_Enzyme_D",
#                "vmax_forward_Enzyme_E","vmax_forward_Enzyme_F","vmax_forward_Enzyme_G"]
#     my_designs=[]
#     cart=[]
#     for i in range(len(sc4)):
#         my_designs.append(dict(zip(enz_names,sc4[i])))
#         cart.append(tuple(sc4[i]))
#     return my_designs,cart  


