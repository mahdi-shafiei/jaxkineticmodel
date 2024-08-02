import tellurium as te
import time
import numpy as np
import matplotlib.pyplot as plt

filepath = (
    "models/sbml_models/"
    "working_models/Garde2020.xml"
    # "working_models/Bertozzi2020.xml"
)

model=te.loadSBMLModel(filepath)
sol=model.simulate(0,100,1000)
# b=time.time()
model.plot()

plt.show()