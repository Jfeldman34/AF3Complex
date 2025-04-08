#!/bin/bash
#
# This is an example script for predicting protein structures with AF3Complex.
# The example is H1233 from CASP16.

# Please first:
# 1) Install AF3Complex, per the installation documentation. 
# 2) Change the following to point to your database directory. 
export DB_DIR="path/to/db"

### specify inputs
model_weights_path="path/to/weights" # the path to the installed AF3 weights. 
output_dir_path="path/to/output"     # the path to the output dir. 
json_file_path="path/to/input.json"  # the path to the input json file.

echo "The path to the model weights is $model_weights_path"
echo "The path to the output directory is $output_dir_path"
echo "The path to the input json file is $json_file_path"

python run_af3complex.py \
  --json_file_path=$json_file_path \
  --model_dir=$model_weights_path \
  --db_dir=$DB_DIR \
  --output_dir=$output_dir_path \
  --input_json_type='af3'
