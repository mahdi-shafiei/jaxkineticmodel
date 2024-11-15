import pandas as pd
import numpy as np
import diffrax
import jax
import jax.numpy as jnp
from models.manual_implementations.glycolysis.glycolysis_model import glycolysis,NeuralODE


def overwrite_y0_dict(y0_dict, dataset):
    """Overwrites the y0 dictionary with the values from the dataset"""
    ynew_dict = y0_dict.copy()
    dataset_iv = dataset.to_dict()

    for met in ynew_dict.keys():
        if met in dataset_iv.keys():
            if not np.isnan(dataset_iv[met]):
                ynew_dict[met] = dataset_iv[met]

    return ynew_dict


def prepare_glycolysis_model(dilution_rate, y0_dict, data_type="glucose_pulse"):
    """"""
    metabolite_names = list(y0_dict.keys())
    filepath = "datasets/VanHeerden_Glucose_Pulse/"
    if data_type == "steady_state":
        dataset_ss = "canelas_SS_data.csv"
        dataset_ss = pd.read_csv(filepath + dataset_ss, index_col=0, header=1)
        dataset_ss = dataset_ss[dilution_rate].T

        y0_dict_new = overwrite_y0_dict(y0_dict, dataset_ss)

        y0 = jnp.array(list(y0_dict_new.values()))

        dataset = pd.DataFrame(jnp.tile(y0, (300, 1)), columns=metabolite_names)
        dataset["ECglucose"] = jnp.ones(300) * dataset_ss["Cs"]

        time_points = jnp.linspace(0, 1000, 300)
        coeffs_ECglucose4 = diffrax.backward_hermite_coefficients(
            ts=jnp.array(time_points), ys=jnp.array(dataset["ECglucose"]), fill_forward_nans_at_end=True
        )
        EC_glucose_interpolation_cubic = diffrax.CubicInterpolation(ts=jnp.array(time_points), coeffs=coeffs_ECglucose4)
        interpolated_mets = {"ECglucose": EC_glucose_interpolation_cubic}

        ECbiomass = [
            3.578710644677661,
            3.7188905547226385,
            3.7683658170914542,
            3.7518740629685157,
            3.001499250374813,
            2.795352323838081,
            1.8470764617691149,
            1.4430284857571216,
        ]
        D = [0.01, 0.05, 0.1, 0.2, 0.3, 0.325, 0.35, 0.375]
        ECbiomass_coeffs = diffrax.backward_hermite_coefficients(ts=jnp.array(D), ys=jnp.array(ECbiomass))
        ECbiomass_interpolated = diffrax.CubicInterpolation(ts=jnp.array(D), coeffs=ECbiomass_coeffs)
        interpolated_mets["ECbiomass"] = ECbiomass_interpolated

        glycolyse = jax.jit(NeuralODE(glycolysis(interpolated_mets, metabolite_names, dilution_rate=float(dilution_rate))))

        dataset = dataset.drop(labels="ECglucose", axis=1)
        # dataset_ss=jnp.array(dataset)

        # dilution_rate=pd.DataFrame
    elif data_type == "glucose_pulse":
        dataset = "vHeerden_trehalose_data_formatted.csv"
        dataset = pd.read_csv(filepath + dataset, index_col=0)

        initial_values_dataset = dataset.iloc[0, :]

        y0_dict_new = overwrite_y0_dict(y0_dict, initial_values_dataset)
        y0 = jnp.array(list(y0_dict_new.values()))

        time_points = [float(i) for i in dataset.index.to_list()]

        # interpolate glucose extracellular
        dataset["ECglucose"] = 110
        coeffs_ECglucose = diffrax.backward_hermite_coefficients(
            ts=jnp.array(time_points), ys=jnp.array(dataset["ECglucose"]), fill_forward_nans_at_end=True
        )
        EC_glucose_interpolation_cubic = diffrax.CubicInterpolation(ts=jnp.array(time_points), coeffs=coeffs_ECglucose)

        interpolated_mets = {"ECglucose": EC_glucose_interpolation_cubic}

        # interpolate extracellular biomass
        ECbiomass = [
            3.578710644677661,
            3.7188905547226385,
            3.7683658170914542,
            3.7518740629685157,
            3.001499250374813,
            2.795352323838081,
            1.8470764617691149,
            1.4430284857571216,
        ]
        D = [0.01, 0.05, 0.1, 0.2, 0.3, 0.325, 0.35, 0.375]
        ECbiomass_coeffs = diffrax.backward_hermite_coefficients(ts=jnp.array(D), ys=jnp.array(ECbiomass))
        ECbiomass_interpolated = diffrax.CubicInterpolation(ts=jnp.array(D), coeffs=ECbiomass_coeffs)
        interpolated_mets["ECbiomass"] = ECbiomass_interpolated

        glycolyse = jax.jit(NeuralODE(glycolysis(interpolated_mets, metabolite_names, dilution_rate=float(dilution_rate))))

        # format dataset to include initial conditions and remove core metabolism TCA cycle metabolites
        dataset.loc[0.00000, "ICglyc"] = y0_dict["ICglyc"]
        dataset.loc[0.00000, "ICBPG"] = y0_dict["ICBPG"]
        dataset.loc[0.00000, "ICNAD"] = y0_dict["ICNAD"]
        dataset.loc[0.00000, "ICNADH"] = y0_dict["ICNADH"]
        dataset.loc[0.00000, "ICACE"] = y0_dict["ICACE"]
        dataset.loc[0.00000, "ICETOH"] = y0_dict["ICETOH"]
        dataset.loc[0.00000, "ICPHOS"] = y0_dict["ICPHOS"]
        dataset.loc[0.00000, "ICIMP"] = y0_dict["ICINO"]
        dataset.loc[0.00000, "ICINO"] = y0_dict["ICINO"]
        dataset.loc[0.00000, "ICHYP"] = y0_dict["ICHYP"]
        dataset.loc[0.00000, "IC2PG"] = y0_dict["IC2PG"]
        dataset.loc[0.00000, "ICglucose"] = y0_dict["ICglucose"]
        dataset.loc[0.00000, "ICDHAP"] = y0_dict["ICDHAP"]
        dataset.loc[0.00000, "ICPYR"] = y0_dict["ICPYR"]
        dataset.loc[0.00000, "ICG3P"] = y0_dict["ICG3P"]
        dataset.loc[0.0000, "ECETOH"] = y0_dict["ECETOH"]
        dataset.loc[0.0000, "ECglycerol"] = y0_dict["ECglycerol"]
        dataset = dataset.drop(labels="ECglucose", axis=1)
        dataset = dataset[metabolite_names]

    return glycolyse, time_points, y0_dict_new, dataset


def update_parameters_by_dilution_rate(params, interpolation_expression_dict, D):
    newparams = params.copy()
    # updates parameters by the dilution rate dependency
    newparams["p_HXK_Vmax"] = interpolation_expression_dict["expr_interpolated_HXK"].evaluate(D) * params["p_HXK_Vmax"]
    newparams["p_PGI1_Vmax"] = interpolation_expression_dict["expr_interpolated_PGI"].evaluate(D) * params["p_PGI1_Vmax"]
    newparams["p_PFK_Vmax"] = interpolation_expression_dict["expr_interpolated_PFK"].evaluate(D) * params["p_PFK_Vmax"]
    newparams["p_FBA1_Vmax"] = interpolation_expression_dict["expr_interpolated_FBA"].evaluate(D) * params["p_FBA1_Vmax"]
    newparams["p_TPI1_Vmax"] = interpolation_expression_dict["expr_interpolated_TPI"].evaluate(D) * params["p_TPI1_Vmax"]
    newparams["p_GAPDH_Vmax"] = interpolation_expression_dict["expr_interpolated_GAPDH"].evaluate(D) * params["p_GAPDH_Vmax"]
    newparams["p_PGK_VmPGK"] = interpolation_expression_dict["expr_interpolated_PGK"].evaluate(D) * params["p_PGK_VmPGK"]
    newparams["p_PGM1_Vmax"] = interpolation_expression_dict["expr_interpolated_PGM"].evaluate(D) * params["p_PGM1_Vmax"]
    newparams["p_ENO1_Vm"] = interpolation_expression_dict["expr_interpolated_ENO"].evaluate(D) * params["p_ENO1_Vm"]
    newparams["p_PYK1_Vm"] = interpolation_expression_dict["expr_interpolated_PYK"].evaluate(D) * params["p_PYK1_Vm"]
    newparams["p_PDC1_Vmax"] = interpolation_expression_dict["expr_interpolated_PDC"].evaluate(D) * params["p_PDC1_Vmax"]
    newparams["p_ADH_VmADH"] = interpolation_expression_dict["expr_interpolated_ADH"].evaluate(D) * params["p_ADH_VmADH"]
    return newparams


def divide_parameters_by_dilution_rate(params, interpolation_expression_dict, D):
    newparams = params.copy()
    # updates parameters by the dilution rate dependency
    newparams["p_HXK_Vmax"] = params["p_HXK_Vmax"] / interpolation_expression_dict["expr_interpolated_HXK"].evaluate(D)
    newparams["p_PGI1_Vmax"] = params["p_PGI1_Vmax"] / interpolation_expression_dict["expr_interpolated_PGI"].evaluate(D)
    newparams["p_PFK_Vmax"] = params["p_PFK_Vmax"] / interpolation_expression_dict["expr_interpolated_PFK"].evaluate(D)
    newparams["p_FBA1_Vmax"] = params["p_FBA1_Vmax"] / interpolation_expression_dict["expr_interpolated_FBA"].evaluate(D)
    newparams["p_TPI1_Vmax"] = params["p_TPI1_Vmax"] / interpolation_expression_dict["expr_interpolated_TPI"].evaluate(D)
    newparams["p_GAPDH_Vmax"] = params["p_GAPDH_Vmax"] / interpolation_expression_dict["expr_interpolated_GAPDH"].evaluate(D)
    newparams["p_PGK_VmPGK"] = params["p_PGK_VmPGK"] / interpolation_expression_dict["expr_interpolated_PGK"].evaluate(D)
    newparams["p_PGM1_Vmax"] = params["p_PGM1_Vmax"] / interpolation_expression_dict["expr_interpolated_PGM"].evaluate(D)
    newparams["p_ENO1_Vm"] = params["p_ENO1_Vm"] / interpolation_expression_dict["expr_interpolated_ENO"].evaluate(D)
    newparams["p_PYK1_Vm"] = params["p_PYK1_Vm"] / interpolation_expression_dict["expr_interpolated_PYK"].evaluate(D)
    newparams["p_PDC1_Vmax"] = params["p_PDC1_Vmax"] / interpolation_expression_dict["expr_interpolated_PDC"].evaluate(D)
    newparams["p_ADH_VmADH"] = params["p_ADH_VmADH"] / interpolation_expression_dict["expr_interpolated_ADH"].evaluate(D)
    return newparams


def load_model_glucose_pulse_FF_datasets(dataset_name, dilution_rate, y0_dict):
    """Loading specific datasets Specific datasets of the ff1 kind"""

    metabolite_names = list(y0_dict.keys())
    filepath = "datasets/VanHeerden_Glucose_Pulse/"
    dataset = pd.read_csv(filepath + dataset_name, index_col=0).T

    initial_values_dataset = dataset.iloc[0, :]

    y0_dict_new = overwrite_y0_dict(y0_dict, initial_values_dataset)


    time_points = [float(i) for i in dataset.index.to_list()]

    coeffs_ECglucose = diffrax.backward_hermite_coefficients(
        ts=jnp.array(time_points), ys=jnp.array(dataset["ECglucose"]), fill_forward_nans_at_end=True
    )
    EC_glucose_interpolation_cubic = diffrax.CubicInterpolation(ts=jnp.array(time_points), coeffs=coeffs_ECglucose)
    interpolated_mets = {"ECglucose": EC_glucose_interpolation_cubic}

    ECbiomass = [
        3.578710644677661,
        3.7188905547226385,
        3.7683658170914542,
        3.7518740629685157,
        3.001499250374813,
        2.795352323838081,
        1.8470764617691149,
        1.4430284857571216,
    ]
    D = [0.01, 0.05, 0.1, 0.2, 0.3, 0.325, 0.35, 0.375]
    ECbiomass_coeffs = diffrax.backward_hermite_coefficients(ts=jnp.array(D), ys=jnp.array(ECbiomass))
    ECbiomass_interpolated = diffrax.CubicInterpolation(ts=jnp.array(D), coeffs=ECbiomass_coeffs)
    interpolated_mets["ECbiomass"] = ECbiomass_interpolated
    glycolyse = jax.jit(NeuralODE(glycolysis(interpolated_mets, metabolite_names, dilution_rate=float(dilution_rate))))
    # format dataset to include initial conditions and remove core metabolism TCA cycle metabolites

    dataset["ICglyc"] = np.nan  # Add column with NaNs
    dataset.iloc[0, dataset.columns.get_loc("ICglyc")] = y0_dict["ICglyc"]  # Set first value

    dataset["ICBPG"] = np.nan
    dataset.iloc[0, dataset.columns.get_loc("ICBPG")] = y0_dict["ICBPG"]

    dataset["ICNAD"] = np.nan
    dataset.iloc[0, dataset.columns.get_loc("ICNAD")] = y0_dict["ICNAD"]

    dataset["ICNADH"] = np.nan
    dataset.iloc[0, dataset.columns.get_loc("ICNADH")] = y0_dict["ICNADH"]

    dataset["ICACE"] = np.nan
    dataset.iloc[0, dataset.columns.get_loc("ICACE")] = y0_dict["ICACE"]

    dataset["ICETOH"] = np.nan
    dataset.iloc[0, dataset.columns.get_loc("ICETOH")] = y0_dict["ICETOH"]

    dataset["ICPHOS"] = np.nan
    dataset.iloc[0, dataset.columns.get_loc("ICPHOS")] = y0_dict["ICPHOS"]

    dataset["ICIMP"] = np.nan
    dataset.iloc[0, dataset.columns.get_loc("ICIMP")] = y0_dict["ICINO"]

    dataset["ICINO"] = np.nan
    dataset.iloc[0, dataset.columns.get_loc("ICINO")] = y0_dict["ICINO"]

    dataset["ICHYP"] = np.nan
    dataset.iloc[0, dataset.columns.get_loc("ICHYP")] = y0_dict["ICHYP"]

    dataset["IC2PG"] = np.nan
    dataset.iloc[0, dataset.columns.get_loc("IC2PG")] = y0_dict["IC2PG"]

    dataset["ICglucose"] = np.nan
    dataset.iloc[0, dataset.columns.get_loc("ICglucose")] = y0_dict["ICglucose"]

    dataset["ICDHAP"] = np.nan
    dataset.iloc[0, dataset.columns.get_loc("ICDHAP")] = y0_dict["ICDHAP"]

    dataset["ICPYR"] = np.nan
    dataset.iloc[0, dataset.columns.get_loc("ICPYR")] = y0_dict["ICPYR"]

    dataset["ICG3P"] = np.nan
    dataset.iloc[0, dataset.columns.get_loc("ICG3P")] = y0_dict["ICG3P"]

    dataset["ECETOH"] = np.nan
    dataset.iloc[0, dataset.columns.get_loc("ECETOH")] = y0_dict["ECETOH"]

    dataset["ECglycerol"] = np.nan
    dataset.iloc[0, dataset.columns.get_loc("ECglycerol")] = y0_dict["ECglycerol"]

    dataset = dataset.drop(labels="ECglucose", axis=1)
    dataset = dataset[metabolite_names]
    return glycolyse, time_points, y0_dict_new, dataset
