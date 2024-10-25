import matplotlib.pyplot as plt
import numpy as np
import itertools
from matplotlib import colors
import matplotlib.patches as mpatches

def plot_probability_distribution(probability_matrix,params):
    """Heatmap of the promoter probability distribution from the DBTL cycle"""
    fig, ax = plt.subplots()
    plt.imshow(probability_matrix,cmap="Reds")
    ax.set_xticks(np.arange(len(params['enz_names'])))
    ax.set_xticklabels(labels=params['enz_names'])
    plt.setp(ax.get_xticklabels(), rotation=45,ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(np.shape(probability_matrix)[0]):
        for j in range(len(params['enz_names'])):
            text = ax.text(j, i, np.round(probability_matrix[i, j],1),
                        ha="center", va="center", color="black")
    plt.colorbar()


def metabolic_engineering_vis_lancet(training_df,params,percentage):
    col_promoters= ['#00468BFF','#ED0000FF','#42B540FF','#0099B4FF',
    '#925E9FFF','#FDAF91FF','#AD002AFF','#ADB6B6FF',
    '#1B1919FF','#00468B99','#ED000099','#42B54099']
    col_promoters= col_promoters
    cmap = colors.ListedColormap(col_promoters)

    all_perturbation_values=[]
    for i in range(len(params['perturb_range'])):
        for j in params['perturb_range'][i]:
            all_perturbation_values.append(j)
    all_perturbation_values=np.unique(all_perturbation_values)
    bounds=np.array(all_perturbation_values)-0.01
    fluxes=training_df[params['engineering_target']]
    designs=training_df[params['enz_names']]
    designs=np.array(designs)
    designs=designs.T
    error=np.random.uniform(low=fluxes*(1-percentage),high=fluxes*(1+percentage))-fluxes
    norm = colors.BoundaryNorm(bounds, cmap.N)
    print(norm)
    labels=np.unique(designs)
    fig1, axs = plt.subplots(figsize=(15,2))
    fig1.tight_layout()

    plt.bar(np.arange(0,np.shape(designs)[1],1),fluxes,yerr=error,color="grey",width=0.6)
    plt.axhline(y=1,c="black",linestyle="--")
    axs.set_xticks(np.arange(0, np.shape(designs)[1], 1))
    axs.set_xticklabels([])
    axs.set_yticklabels([])
    plt.ylabel("Rel. flux")

    fig, axs = plt.subplots(figsize=(15,15))
    # #Plot 1
    axs= plt.imshow(designs,cmap=cmap,norm=norm)
    axs= plt.gca()

    axs.set_xticks(np.arange(0, np.shape(designs)[1], 1))
    axs.set_yticks(np.arange(0, np.shape(designs)[0], 1))
    # # # Labels for major ticks
    enz=params['enz_names']
    axs.set_yticklabels(enz,rotation=0)
    axs.set_xticklabels([])
    # # Minor ticks
    axs.set_xticks(np.arange(-.5, np.shape(designs)[1],1), minor=True)
    axs.set_yticks(np.arange(-.5, np.shape(designs)[0], 1), minor=True)
    # # Gridlines based on minor ticks
    axs.grid(which='minor', color="w", linestyle='-', linewidth=2)


    legend_values=[]
    for i in range(len(col_promoters)):
        one=mpatches.Patch(color=col_promoters[i], label=all_perturbation_values[i])
        legend_values.append(one)
    plt.legend(handles=legend_values,title="Element strength",bbox_to_anchor=(1.14, 1.0))
    return fig1,fig, axs

