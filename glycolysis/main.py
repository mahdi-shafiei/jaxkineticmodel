#data prep + create fluxes + call trainer
from fluxes import create_fluxes
from glycolysis import *

import torch
from torchvision import models
from torchsummary import summary
def main():
    fluxes = create_fluxes()
    print(fluxes)
  
    metabolites = {}
    glycolysis = Glycolysis(fluxes=fluxes, metabolites=metabolites)

if __name__ == "__main__":
    main()