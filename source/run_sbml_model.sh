
# Bash script for the sbml model clustering
directory_path="../parameter_initializations/simple_sbml_initializations/lhs/"
bounds_file="../parameter_initializations/simple_sbml_initializations/simple_sbml_bounds.csv"
method="lhs"
model="../models/SBML_models/simple_sbml.xml"
name="simple_sbml_run1" 
output_dir="../results/simple_sbml/lhs/"
data="../data/rawdata_simple_sbml_p1.csv"
nparameters=10
divide=1
python_file=cluster_sbml_test.py

python3 initialize_parameters.py -n $name -f $bounds_file -m $method -s $nparameters -d $divide -o $directory_path

pwd
# Check if the directory exists
if [ -d "$directory_path" ]; then
    # Loop over files in the directory
    for file in "$directory_path"/*; do
        # Check if the current item is a file
        if [ -f "$file" ]; then
            # Echo the file name
            echo "File: $file"
            echo "Python: $python_file"
            echo "outputdir:$output_dir"
            python3 $python_file -n $name -m $model -p $file -d $data -o $output_dir 
        fi
    done
else
    echo "Directory not found: $directory_path"
fi

echo $file

