import tellurium as te
import time
import numpy as np
import matplotlib.pyplot as plt

filepath = (
    "models/sbml_models/discrepancies/Lucarelli_CellSystems_2018.xml"
    # "working_models/Bertozzi2020.xml"
)

model=te.loadSBMLModel(filepath)
sol=model.simulate(0,5,1000)
# b=time.time()




model.plot()
plt.show()


filepath = (
    "models/sbml_models/discrepancies/Lucarelli_CellSystems_2018.xml"
    # "working_models/Bertozzi2020.xml"
)

model=te.loadSBMLModel(filepath)
sol=model.simulate(0,15,1000)
# b=time.time()




model.plot()
