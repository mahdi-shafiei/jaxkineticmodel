
# Bash script for the batch bioprocess, monod model, and other self-implemented models.



directory_path="../parameter_initializations/pCA_fermentation_initializations/lhs/"

pwd
bounds_file="../parameter_initializations/pCA_fermentation_initializations/pCA_fermentation_bounds.csv"
method="lhs"
name="pca_run1" 
output_dir="../results/pca_fermentation/lhs/"
data="../data/pCA_timeseries/pCA_fermentation_data_200424.csv"
nparameters=1
divide=1
python_file=cluster_pCA_model.py



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
            python3 $python_file -n $name -p $file -d $data -o $output_dir 
        fi
    done
else
    echo "Directory not found: $directory_path"
fi

echo $file

