# Set the directory path

directory_path=~/Documenten/Gitlab/NeuralODEs/glycolysis/parameter_initializations/monod_initializations/lhs

python3 initialize_parameters_monod.py

# Check if the directory exists
if [ -d "$directory_path" ]; then
    # Loop over files in the directory
    for file in "$directory_path"/*; do
        # Check if the current item is a file
        if [ -f "$file" ]; then
            # Echo the file name
            echo "File: $file"
            python3 cluster_monod.py -p $file 
        fi
    done
else
    echo "Directory not found: $directory_path"
fi

echo $file

