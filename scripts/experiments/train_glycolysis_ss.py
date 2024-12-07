import sys

sys.path.append("/tudelft.net/staff-bulk/ewi/insy/DBL/plent/NeuralODEs/jax_neural_odes")
sys.path.append("/home/plent/Documenten/Gitlab/NeuralODEs/jax_neural_odes")
import pandas as pd
import numpy as np
import jax.numpy as jnp
import jax
import optax
from models.manual_implementations.glycolysis.glycolysis_model import *

jax.config.update("jax_enable_x64", True)
from jaxkineticmodel.parameter_estimation.training import (
    create_log_params_means_centered_loss_func2,
    log_transform_parameters,
    exponentiate_parameters,
)
from scripts.analysis_helper_functions.helper_function_glycolysis_analysis import (
    prepare_glycolysis_model,
    update_parameters_by_dilution_rate,
)
from jaxkineticmodel.utils import get_logger

logger = get_logger(__name__)


expression_data = pd.read_csv("datasets/VanHeerden_Glucose_Pulse/PvanHoekExpressionData.csv", index_col=0)

column_pairs = [
    ("D_HXK", "HXK"),
    ("D_PGI", "PGI"),
    ("D_PFK", "PFK"),
    ("D_FBA", "FBA"),
    ("D_TPI", "TPI"),
    ("D_GAPDH", "GAPDH"),
    ("D_PGK", "PGK"),
    ("D_PGM", "PGM"),
    ("D_ENO", "ENO"),
    ("D_PYK", "PYK"),
    ("D_PDC", "PDC"),
    ("D_ADH", "ADH"),
]

interpolation_expression_dict = {}

Ds = jnp.linspace(0, 0.375, 100)

for D_col, col in column_pairs:
    interp_key = f"expr_interpolated_{col}"
    interpolation_expression_dict[interp_key] = diffrax.LinearInterpolation(
        ts=jnp.array(expression_data[D_col]), ys=jnp.array(expression_data[col])
    )


y0_dict = {
    "ICG1P": 0.064568,
    "ICT6P": 0.093705,
    "ICtreh": 63.312040,
    "ICglucose": 0.196003,
    "ICG6P": 0.716385,
    "ICF6P": 0.202293,
    "ICFBP": 0.057001,
    "ICDHAP": 0.048571,
    "ICG3P": 0.020586,
    "ICglyc": 0.1,
    "ICGAP": 0.006213,
    "ICBPG": 0.0001,
    "IC3PG": 2.311074,
    "IC2PG": 0.297534,
    "ICPEP": 1.171415,
    "ICPYR": 0.152195,
    "ICACE": 0.04,
    "ICETOH": 10.0,
    "ECETOH": 0,
    "ECglycerol": 0.0,
    "ICNADH": 0.0106,
    "ICNAD": 1.5794,
    "ICATP": 3.730584,
    "ICADP": 1.376832,
    "ICAMP": 0.431427,
    "ICPHOS": 10,
    "ICIMP": 0.100,
    "ICINO": 0.100,
    "ICHYP": 1.5,
}
y0 = jnp.array(list(y0_dict.values()))
metabolite_names = list(y0_dict.keys())

# glycolyse=glycolysis(interpolated_mets,metabolite_names,dilution_rate=0.1)
# term=diffrax.ODETerm(glycolyse)
# dataset1=pd.read_csv("datasets/VanHeerden_Glucose_Pulse/FF1_timeseries_format.csv",index_col=0).T
# time_points=[int(i) for i in dataset1.index.to_list()]

# glycolyse_GP1,time_points_GP1,y0_GP1,dataset_GP1=prepare_glycolysis_model(data_type="glucose_pulse",dilution_rate="0.1",y0_dict=y0_dict)
glycolyse_SS_01, time_points_SS_01, y0_SS_01, dataset_SS_01 = prepare_glycolysis_model(
    data_type="steady_state", dilution_rate="0.02", y0_dict=y0_dict
)
glycolyse_SS_05, time_points_SS_05, y0_SS_05, dataset_SS_05 = prepare_glycolysis_model(
    data_type="steady_state", dilution_rate="0.05", y0_dict=y0_dict
)
glycolyse_SS_10, time_points_SS_10, y0_SS_10, dataset_SS_10 = prepare_glycolysis_model(
    data_type="steady_state", dilution_rate="0.1", y0_dict=y0_dict
)
glycolyse_SS_20, time_points_SS_20, y0_SS_20, dataset_SS_20 = prepare_glycolysis_model(
    data_type="steady_state", dilution_rate="0.2", y0_dict=y0_dict
)
glycolyse_SS_30, time_points_SS_30, y0_SS_30, dataset_SS_30 = prepare_glycolysis_model(
    data_type="steady_state", dilution_rate="0.3", y0_dict=y0_dict
)
glycolyse_SS_325, time_points_SS_325, y0_SS_325, dataset_SS_325 = prepare_glycolysis_model(
    data_type="steady_state", dilution_rate="0.325", y0_dict=y0_dict
)
glycolyse_SS_35, time_points_SS_35, y0_SS_35, dataset_SS_35 = prepare_glycolysis_model(
    data_type="steady_state", dilution_rate="0.35", y0_dict=y0_dict
)
glycolyse_SS_375, time_points_SS_375, y0_SS_375, dataset_SS_375 = prepare_glycolysis_model(
    data_type="steady_state", dilution_rate="0.375", y0_dict=y0_dict
)

logger.info("Datasets and models properly loaded")


datasets = {
    "SS_01": jnp.array(dataset_SS_01),
    "SS_05": jnp.array(dataset_SS_05),
    "SS_10": jnp.array(dataset_SS_10),
    "SS_20": jnp.array(dataset_SS_20),
    "SS_30": jnp.array(dataset_SS_30),
    "SS_325": jnp.array(dataset_SS_325),
    "SS_35": jnp.array(dataset_SS_35),
    "SS_375": jnp.array(dataset_SS_375),
}

time_points = {
    "SS_01": jnp.array(time_points_SS_01),
    "SS_05": jnp.array(time_points_SS_05),
    "SS_10": jnp.array(time_points_SS_10),
    "SS_20": jnp.array(time_points_SS_20),
    "SS_30": jnp.array(time_points_SS_30),
    "SS_325": jnp.array(time_points_SS_325),
    "SS_35": jnp.array(time_points_SS_35),
    "SS_375": jnp.array(time_points_SS_375),
}


# loss_targets_glucose_pulse=[0,1,2,
#                             4,5,6,10,12,14,20,21,22]
loss_targets_steady_state = [4, 5, 6, 7, 10, 12, 13, 14, 15, 17, 20, 21, 22]


log_loss_func_SS_01 = jax.jit(create_log_params_means_centered_loss_func2(glycolyse_SS_01, loss_targets_steady_state))
log_loss_func_SS_05 = jax.jit(create_log_params_means_centered_loss_func2(glycolyse_SS_05, loss_targets_steady_state))
log_loss_func_SS_10 = jax.jit(create_log_params_means_centered_loss_func2(glycolyse_SS_10, loss_targets_steady_state))
log_loss_func_SS_20 = jax.jit(create_log_params_means_centered_loss_func2(glycolyse_SS_20, loss_targets_steady_state))
log_loss_func_SS_30 = jax.jit(create_log_params_means_centered_loss_func2(glycolyse_SS_30, loss_targets_steady_state))
log_loss_func_SS_325 = jax.jit(create_log_params_means_centered_loss_func2(glycolyse_SS_325, loss_targets_steady_state))
log_loss_func_SS_35 = jax.jit(create_log_params_means_centered_loss_func2(glycolyse_SS_35, loss_targets_steady_state))
log_loss_func_SS_375 = jax.jit(create_log_params_means_centered_loss_func2(glycolyse_SS_375, loss_targets_steady_state))


params_literature = pd.read_csv(
    "parameter_initializations/Glycolysis_model/parameter_initialization_glycolysis_literature_values.csv", index_col=0
).to_dict()["0"]


lr = 1e-3

optimizer = optax.adabelief(lr)
clip_by_global = optax.clip_by_global_norm(np.log(4))
optimizer = optax.chain(optimizer, clip_by_global)
params_init = params_literature

opt_state = optimizer.init(params_init)


loss_per_iter1 = []
loss_per_iter2 = []
loss_per_iter3 = []
loss_per_iter4 = []
loss_per_iter5 = []
loss_per_iter6 = []
loss_per_iter7 = []
loss_per_iter8 = []
loss_per_iter9 = []


grads_SS_01 = jax.jit(jax.grad(log_loss_func_SS_01, 0))
grads_SS_05 = jax.jit(jax.grad(log_loss_func_SS_05, 0))
grads_SS_10 = jax.jit(jax.grad(log_loss_func_SS_10, 0))
grads_SS_20 = jax.jit(jax.grad(log_loss_func_SS_20, 0))
grads_SS_30 = jax.jit(jax.grad(log_loss_func_SS_30, 0))
grads_SS_325 = jax.jit(jax.grad(log_loss_func_SS_325, 0))
grads_SS_35 = jax.jit(jax.grad(log_loss_func_SS_35, 0))
grads_SS_375 = jax.jit(jax.grad(log_loss_func_SS_375, 0))

logger.info("Gradient descent can start")

ys = datasets
ts = time_points

for step in range(400):
    grads = {}

    log_params = log_transform_parameters(
        params_init
    )  # not used to calculate gradients, but is updated in line 57 (optax apply updates)
    log_params_01 = log_transform_parameters(
        update_parameters_by_dilution_rate(params_init, interpolation_expression_dict, D=0.01)
    )
    log_params_05 = log_transform_parameters(
        update_parameters_by_dilution_rate(params_init, interpolation_expression_dict, D=0.05)
    )
    log_params_10 = log_transform_parameters(
        update_parameters_by_dilution_rate(params_init, interpolation_expression_dict, D=0.1)
    )
    log_params_20 = log_transform_parameters(
        update_parameters_by_dilution_rate(params_init, interpolation_expression_dict, D=0.2)
    )
    log_params_30 = log_transform_parameters(
        update_parameters_by_dilution_rate(params_init, interpolation_expression_dict, D=0.3)
    )
    log_params_325 = log_transform_parameters(
        update_parameters_by_dilution_rate(params_init, interpolation_expression_dict, D=0.325)
    )
    log_params_35 = log_transform_parameters(
        update_parameters_by_dilution_rate(params_init, interpolation_expression_dict, D=0.35)
    )
    log_params_375 = log_transform_parameters(
        update_parameters_by_dilution_rate(params_init, interpolation_expression_dict, D=0.375)
    )

    ## gradients are calculated for the update parameters

    loss2 = log_loss_func_SS_01(log_params_01, ts["SS_01"], ys["SS_01"])
    grads2 = grads_SS_01(log_params_01, ts["SS_01"], ys["SS_01"])

    loss3 = log_loss_func_SS_05(log_params_05, ts["SS_05"], ys["SS_05"])
    grads3 = grads_SS_05(log_params_05, ts["SS_05"], ys["SS_05"])

    loss4 = log_loss_func_SS_10(log_params_10, ts["SS_10"], ys["SS_10"])
    grads4 = grads_SS_10(log_params_10, ts["SS_10"], ys["SS_10"])

    loss5 = log_loss_func_SS_20(log_params_20, ts["SS_20"], ys["SS_20"])
    grads5 = grads_SS_20(log_params_20, ts["SS_20"], ys["SS_20"])

    loss6 = log_loss_func_SS_30(log_params_30, ts["SS_30"], ys["SS_30"])
    grads6 = grads_SS_30(log_params_30, ts["SS_30"], ys["SS_30"])

    loss7 = log_loss_func_SS_325(log_params_325, ts["SS_325"], ys["SS_325"])
    grads7 = grads_SS_325(log_params_325, ts["SS_325"], ys["SS_325"])

    loss8 = log_loss_func_SS_35(log_params_35, ts["SS_35"], ys["SS_35"])
    grads8 = grads_SS_35(log_params_35, ts["SS_35"], ys["SS_35"])

    loss9 = log_loss_func_SS_375(log_params_375, ts["SS_375"], ys["SS_375"])
    grads9 = grads_SS_375(log_params_375, ts["SS_375"], ys["SS_375"])

    # Gradients are averaged
    for key in grads2.keys():
        grads[key] = (
            grads2[key] + grads3[key] + grads4[key] + grads5[key] + grads6[key] + grads7[key] + grads8[key] + grads9[key]
        ) / 8

    # we perform the updataset
    updates, opt_state = optimizer.update(grads, opt_state)
    # we perform updates in log space, but only return params in lin space
    log_params = optax.apply_updates(log_params, updates)
    lin_params = exponentiate_parameters(log_params)
    params_init = lin_params

    # loss_per_iter1.append(float(loss1))
    loss_per_iter2.append(float(loss2))
    loss_per_iter3.append(float(loss3))
    loss_per_iter4.append(float(loss4))
    loss_per_iter5.append(float(loss5))
    loss_per_iter6.append(float(loss6))
    loss_per_iter7.append(float(loss7))
    loss_per_iter8.append(float(loss8))
    loss_per_iter9.append(float(loss9))

    if step % 20 == 0:
        print(f"Step {step}, Loss {loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9}")


losses_per_iterations = {
    "loss_per_iter2": loss_per_iter2,
    "loss_per_iter3": loss_per_iter3,
    "loss_per_iter4": loss_per_iter4,
    "loss_per_iter5": loss_per_iter5,
    "loss_per_iter6": loss_per_iter6,
    "loss_per_iter7": loss_per_iter7,
    "loss_per_iter8": loss_per_iter8,
    "loss_per_iter9": loss_per_iter9,
}
losses_per_iterations_added = pd.DataFrame(losses_per_iterations)
losses_per_iterations_added.to_csv(
    "results/EXP4_Glycolysis_Fitting_Datasets/1309_losses_ss_datasets_literature_parameter_init.csv"
)
params_to_save = pd.DataFrame(pd.Series(params_init))
params_to_save.to_csv("results/EXP4_Glycolysis_Fitting_Datasets/1309_trained_params_ss_datasets_literature_parameter_init.csv")
