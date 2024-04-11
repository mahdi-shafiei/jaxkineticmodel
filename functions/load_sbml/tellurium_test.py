import tellurium as te

x=te.loadSBMLModel("BIOMD0000000244_url.xml")
# x.simulate(0,10,100)
# x.plot()
x.getNumGlobalParameters()
x.getReactionRates()