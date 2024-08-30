import tellurium as te
import time
import numpy as np
import matplotlib.pyplot as plt

filepath = (
    "models/sbml_models/working_models/Borghans_BiophysChem1997.xml"
    # "working_models/Bertozzi2020.xml"
)

model=te.loadSBMLModel(filepath)
sol=model.simulate(0,5,1000)
# b=time.time()




model.plot()
plt.show()


filepath = (
    "models/sbml_models/working_models/Crauste_CellSystems2017.xml"
    # "working_models/Bertozzi2020.xml"
)

model=te.loadSBMLModel(filepath)
sol=model.simulate(0,15,1000)
# b=time.time()




model.plot()
plt.show()