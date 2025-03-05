"""Script to make pypesto ready for compiling + time comparison of compile"""

import amici
import time
import pandas as pd
import libsbml

sbml_file_dir = "models/manual_implementations/sbml_export"
output_dir = "results/PyPESTO_comparison/"
model_name = "glycolysis_feastfamine_pulse1_pypesto"

sbml_file = f"{sbml_file_dir}/{model_name}.xml"
# Create an SbmlImporter instance for our SBML model
sbml_importer = amici.SbmlImporter(sbml_file)

constant_parameters = ['ECbiomass', 'D']

model_name_spline = "glycolysis_feastfamine_pulse1_pypesto_spline"


sbml_model=sbml_importer.sbml
print(sbml_importer.sbml.getSBMLDocument())
data = pd.read_csv('datasets/VanHeerden_Glucose_Pulse/FF1_timeseries_format.csv', index_col=0)
domain = [float(i) for i in data.loc['ECglucose'].dropna().index]
drange = data.loc['ECglucose'].dropna().values

print(domain)
print(drange)


spline = amici.splines.CubicHermiteSpline(
    sbml_id="ECglucose",
    evaluate_at=amici.sbml_utils.amici_time_symbol,  # the spline function is evaluated at the current time point
    nodes=domain,
    values_at_nodes=drange,)



# only run once
# spline.add_to_sbml_model(sbml_model)

libsbml.writeSBMLToFile(sbml_model.getSBMLDocument(),
                        f"models/manual_implementations/sbml_export/{model_name_spline}.xml")


sbml_file = f"{sbml_file_dir}/{model_name_spline}.xml"
# Create an SbmlImporter instance for our SBML model
sbml_importer = amici.SbmlImporter(sbml_file)

constant_parameters = ['ECbiomass', 'D']
# Retrieve model output names and formulae from AssignmentRules and remove the respective rules
observables = amici.assignmentRules2observables(
    sbml_importer.sbml,  # the libsbml model object
    filter_function=lambda variable: variable.getId() == "ECglucose")
print('observables', observables)


# pypesto requires setting observables. The logic behind is that there is a mapping
# from observations (measurements) to state variables, that might be nonlinear  etc.
#

metab_names=['ICATP', 'ICglucose', 'ICADP', 'ICG6P', 'ICtreh', 'ICF6P', 'ICG1P',
       'ICT6P', 'ICFBP', 'ICGAP', 'ICDHAP', 'ICG3P', 'IC3PG', 'ICglyc', 'IC2PG', 'ICPEP',
       'ICPYR', 'ICAMP']


for metab in metab_names:
    observables[f"obs_{metab}"]={'formula':metab,
                                 'name':f"obs_{metab}"}

print(observables)
start = time.time()
sbml_importer.sbml2amici(
    model_name_spline,
    output_dir,
    verbose=True,
    observables=observables,
    constant_parameters=constant_parameters,
)  #compile step takes about 14181.56623506546
end = time.time()
print(f'compiling took {end - start} seconds')
