import numpy as np

import pandas as pd
import re


def retrieve_convergence_results(results_dir, file_list):
    # file_list

    loss_dfs = {}
    params_dfs = {}
    norms_dfs = {}
    for file in file_list:
        if "losses" in file:
            loss_data = pd.read_csv(results_dir + file, index_col=0)
            loss_dfs[file] = loss_data
        if "parameters" in file:
            params_data = pd.read_csv(results_dir + file, index_col=0)
            params_dfs[file] = params_data
        if "norms" in file:
            norms_data = pd.read_csv(results_dir + file, index_col=0)
            norms_dfs[file] = norms_data
    return loss_dfs, params_dfs, norms_dfs


def retrieve_success_rate(loss_dfs, model_name):
    """retrieve initializatoin success from loss_dfs"""
    initialization_succes_dict = {}
    lb_2 = []
    lb_5 = []
    lb_10 = []
    lb_50 = []
    lb_100 = []
    for bound in list(loss_dfs.keys()):
        if re.search(model_name, bound):
            # print(bound)
            if re.search("bounds_2.csv", bound):
                loss_df = loss_dfs[bound]
                percentage_succeeded_initializations = len(np.where(loss_df.iloc[0, :] != -1)[0]) / np.shape(loss_df)[1] * 100
                lb_2.append(percentage_succeeded_initializations)

            if re.search("bounds_5.csv", bound):
                loss_df = loss_dfs[bound]

                percentage_succeeded_initializations = len(np.where(loss_df.iloc[0, :] != -1)[0]) / np.shape(loss_df)[1] * 100
                lb_5.append(percentage_succeeded_initializations)
            if re.search("bounds_10.csv", bound):
                loss_df = loss_dfs[bound]
                percentage_succeeded_initializations = len(np.where(loss_df.iloc[0, :] != -1)[0]) / np.shape(loss_df)[1] * 100
                lb_10.append(percentage_succeeded_initializations)

            if re.search("bounds_50.csv", bound):
                loss_df = loss_dfs[bound]
                percentage_succeeded_initializations = len(np.where(loss_df.iloc[0, :] != -1)[0]) / np.shape(loss_df)[1] * 100
                lb_50.append(percentage_succeeded_initializations)

            if re.search("bounds_100.csv", bound):
                loss_df = loss_dfs[bound]
                percentage_succeeded_initializations = len(np.where(loss_df.iloc[0, :] != -1)[0]) / np.shape(loss_df)[1] * 100
                lb_100.append(percentage_succeeded_initializations)
            initialization_succes_dict["lb_2"] = lb_2
            initialization_succes_dict["lb_5"] = lb_5
            initialization_succes_dict["lb_10"] = lb_10
            initialization_succes_dict["lb_50"] = lb_50
            initialization_succes_dict["lb_100"] = lb_100

    return initialization_succes_dict
