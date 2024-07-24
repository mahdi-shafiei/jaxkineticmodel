import tellurium as te
import time
import numpy as np
import matplotlib.pyplot as plt

filepath = (
    "models/sbml_models/"
    "working_models/Palani2011.xml"
    # "working_models/Bertozzi2020.xml"
)

model=te.loadSBMLModel(filepath)
sol=model.simulate(0,2000,1000)
# b=time.time()
model.plot()
plt.show()