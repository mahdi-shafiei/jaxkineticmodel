import tellurium as te
import time
import numpy as np
import matplotlib.pyplot as plt



model=te.loadSBMLModel("models/sbml_models/working_models/Berzins2022 - C cohnii glucose and glycerol.xml")
sol=model.simulate(0,12,1000)
# b=time.time()
model.plot()