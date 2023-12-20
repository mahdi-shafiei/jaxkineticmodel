import numpy
from parameter_initializations.sampling_methods import uniform_sampling,latinhypercube_sampling
import os
import argparse
import pandas as pd





def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('-n',"--name",type=str,required=True,help="name used to save the parameter sets")
    parser.add_argument('-f',"--bound_file",type=str,required=True,help="lower and upper bound of parameter file: P rows and 2 columns")
    parser.add_argument('-m',"--method",type=str,required=True,help="method for initial sampling. Choices: lhs and uniform")
    parser.add_argument('-s',"--size",type=int,required=True,help="number of initial guesses that should be considered")
    parser.add_argument('-d',"--divide_size",type=int,required=False,help="used to divide workload. Number of files created is size/divide.")
    parser.add_argument('-o',"--output_dir",type=str,required=True,help="output directory")
    args=parser.parse_args()
    

    N_param_sets=args.size
    divide=args.divide_size
    if divide==None:
        divide= args.size
    output_dir=args.output_dir

    bounds=pd.read_csv(args.bound_file,index_col=0)
    lb=bounds['lb']
    ub=bounds['ub']
    names=list(lb.index)
    bounds=tuple(zip(lb,ub))
    bounds=dict(zip(names,bounds))
    
    if args.method=="lhs":
        parameter_sets=latinhypercube_sampling(bounds,N_param_sets)
    elif args.method=="uniform":
        parameter_sets=uniform_sampling(bounds,N_param_sets)
    else:
        print("Method not known")


    file_string=output_dir+args.name+"_"+args.method+"_"
    for i in range(0,N_param_sets,divide):
        subset=parameter_sets.iloc[i:i+divide,:]
        subset.to_csv(file_string+str(i)+".csv")

if __name__=="__main__":
    main()



    
    


# lb=[-0.5,0.000,-1.8,-0.2]
# ub=[-0.1,0.1,-1.4,0.0]
# names=["qsmax","Ks","a","ms"]

# bounds=tuple(zip(lb,ub))
# bounds=dict(zip(names,bounds))
# parameter_sets=latinhypercube_sampling(bounds,N_param_sets)
# #parameter_sets=uniform_sampling(bounds,N_param_sets)

# # directory="../parameter_initializations/batch_bioprocess_initializations/lhs/"


# for i in range(0,400,divide):
#     subset=parameter_sets.iloc[i:i+divide,:]
#     subset.to_csv(directory+"batch_parameter_sets_"+str(i)+".csv")