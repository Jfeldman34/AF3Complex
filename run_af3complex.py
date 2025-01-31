import json
import subprocess
import tempfile
import fcntl
import os
import shutil
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run AF3Complex.")
    parser.add_argument("--json_file_path", required=True, help="Path to the JSON file containing input data.")
    parser.add_argument("--model_dir", required=True, help="Path to the model directory.")
    parser.add_argument("--db_dir", required=True, help="Path to the database directory.")
    parser.add_argument("--output_dir", required=True, help="Path to the output directory.")
    parser.add_argument("--input_json_type", choices=["af3", "server"], required=True, help="Specify the input JSON type: 'af3' or 'server'.")
    return parser.parse_args()

def get_processing_file_path(json_file_path):
    """Returns the processing file path in the same directory as the input JSON file."""
    json_dir = os.path.dirname(json_file_path)
    return os.path.join(json_dir, "processing_file.txt")

def add_to_processing(processing_file, object_name):
    """
    Adds the protein name to the shared processing list with file locking.
    """
    with open(processing_file, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX) 
        f.seek(0)
        current_objects = f.read().splitlines()
        if object_name not in current_objects:
            f.write(object_name + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)  
        print(f"Processing {object_name}")

def remove_from_processing(processing_file, object_name):
    """
    Removes the object name from the shared processing list with file locking.
    """
    try:
        with open(processing_file, "r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)  
            current_objects = f.read().splitlines()
            f.seek(0)
            f.truncate()  
            for obj in current_objects:
                if obj != object_name:
                    f.write(obj + "\n")
            fcntl.flock(f, fcntl.LOCK_UN) 
    except FileNotFoundError:
        pass  

def is_in_processing(processing_file, object_name):
    """
    Checks if an object is already in the shared processing list.
    """
    try:
        with open(processing_file, "r") as f:
            fcntl.flock(f, fcntl.LOCK_SH) 
            current_objects = f.read().splitlines()
            fcntl.flock(f, fcntl.LOCK_UN) 
            if object_name in current_objects:
                print(f"{object_name} already in processing. Skipping...")
            return object_name in current_objects
    except FileNotFoundError:
        return False  

def load_json_objects(file_path):
    """Loads a JSON file containing a list of JSON objects and return it as a list."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def main():
    args = parse_arguments()

    json_file_path = args.json_file_path
    model_dir = args.model_dir
    db_dir = args.db_dir
    output_dir = args.output_dir
    input_json_type = args.input_json_type

    processing_file = get_processing_file_path(json_file_path)

    for individual_json in load_json_objects(json_file_path):
        protein_id = individual_json['name']
        output_dir_check = os.path.join(output_dir, individual_json['name'])
        output_lower_dir_check = os.path.join(output_dir, individual_json['name'].lower())

        if os.path.isdir(output_dir_check) or os.path.isdir(output_lower_dir_check) or is_in_processing(processing_file, individual_json['name']):
            print(f"A model has already been generated for {individual_json['name']}")
            continue

        sequences = individual_json.get('sequences', [])
        contains_ligand = any('ligand' in seq for seq in sequences)
        print(f"Contains ligands: {contains_ligand}")

        # Create a temporary JSON file for the individual JSON object
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            if input_json_type == ligand: 
                json.dump([individual_json], temp_file)
            else: 
                json.dump(individual_json, temp_file)

            temp_file_path = temp_file.name

        command = [
            "python", "run_intermediate.py",
            f"--json_path={temp_file_path}",
            f"--model_dir={model_dir}",
            f"--db_dir={db_dir}",
            f"--output_dir={output_dir}"
        ]

        try:
            add_to_processing(processing_file, individual_json['name'])
            subprocess.run(command, check=True)
            print(f"First model successfully generated for {individual_json['name']}")

            if contains_ligand:
                protein_folder = os.path.join(output_dir, protein_id.lower())
                data_json_path = os.path.join(protein_folder, f"{protein_id.lower()}_data.json")
                if os.path.exists(data_json_path):
                    with open(data_json_path, 'r') as data_file:
                        new_json = json.load(data_file)

                    new_json['name'] = f"{new_json['name']}_without_ligands"
                    new_sequences = [seq for seq in new_json.get('sequences', []) if 'ligand' not in seq]
                    new_json['sequences'] = new_sequences

                    # Create a temporary modified JSON file
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as new_temp_file:
                        if input_json_type == ligand: 
                            json.dump([individual_json], temp_file)
                        else: 
                            json.dump(individual_json, temp_file)
                        new_temp_file_path = new_temp_file.name

                    print(f"Generating a secondary model for {individual_json['name']}")

                    second_command = [
                        "python", "run_intermediate.py",
                        f"--json_path={new_temp_file_path}",
                        f"--model_dir={model_dir}",
                        f"--db_dir={db_dir}",
                        f"--output_dir={output_dir}"
                    ]
                    subprocess.run(second_command, check=True)
                    print(f"Second model successfully generated for {individual_json['name']}")
                    os.remove(new_temp_file_path)
        except subprocess.CalledProcessError as e:
            print(f"Error running the AlphaFold intermediary script for {individual_json['name']}: {e}")
        finally:
            os.remove(temp_file_path)
            remove_from_processing(processing_file, individual_json['name'])

if __name__ == "__main__":
    main()
