import json
import subprocess
import tempfile
import fcntl
import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run AF3Complex Feature Generation.")
    parser.add_argument("--json_file_path", required=True, help="Path to the JSON file containing input data.")
    parser.add_argument("--model_dir", required=True, help="Path to the model directory.")
    parser.add_argument("--db_dir", required=True, help="Path to the database directory.")
    parser.add_argument("--feature_dir", required=True, help="Path to the output directory that will contain model features.")
    parser.add_argument("--input_json_type", choices=["af3", "server"], required=True, help="Specify the input JSON type: 'af3' or 'server'.")
    return parser.parse_args()

def get_processing_file_path(json_file_path):
    json_dir = os.path.dirname(json_file_path)
    return os.path.join(json_dir, "feature_processing_file.txt")

def get_lock_file_path(processing_file):
    return processing_file + ".lock"

def add_to_processing(processing_file, object_name):
    lock_file_path = get_lock_file_path(processing_file)
    with open(lock_file_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        os.makedirs(os.path.dirname(processing_file), exist_ok=True)
        if not os.path.exists(processing_file):
            open(processing_file, 'w').close()
        with open(processing_file, "r+") as f:
            current_objects = f.read().splitlines()
            if object_name not in current_objects:
                f.write(object_name + "\n")
        fcntl.flock(lock, fcntl.LOCK_UN)
        print(f"Processing {object_name}")

def remove_from_processing(processing_file, object_name):
    lock_file_path = get_lock_file_path(processing_file)
    try:
        with open(lock_file_path, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not os.path.exists(processing_file):
                return
            with open(processing_file, "r+") as f:
                current_objects = f.read().splitlines()
                f.seek(0)
                f.truncate()
                for obj in current_objects:
                    if obj != object_name:
                        f.write(obj + "\n")
            fcntl.flock(lock, fcntl.LOCK_UN)
    except FileNotFoundError:
        pass

def is_in_processing(processing_file, object_name):
    lock_file_path = get_lock_file_path(processing_file)
    try:
        with open(lock_file_path, "r") as lock:
            fcntl.flock(lock, fcntl.LOCK_SH)
            if not os.path.exists(processing_file):
                return False
            with open(processing_file, "r") as f:
                current_objects = f.read().splitlines()
            fcntl.flock(lock, fcntl.LOCK_UN)
            if object_name in current_objects:
                print(f"{object_name} already in processing. Skipping...")
            return object_name in current_objects
    except FileNotFoundError:
        return False

def load_json_objects(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data if isinstance(data, list) else [data]

def main():
    args = parse_arguments()

    json_file_path = args.json_file_path
    model_dir = args.model_dir
    db_dir = args.db_dir
    output_dir = args.feature_dir
    input_json_type = args.input_json_type

    processing_file = get_processing_file_path(json_file_path)

    for individual_json in load_json_objects(json_file_path):
        protein_id = individual_json['name']
        output_dir_check = os.path.join(output_dir, protein_id)
        output_lower_dir_check = os.path.join(output_dir, protein_id.lower())

        if os.path.isdir(output_dir_check) or os.path.isdir(output_lower_dir_check) or is_in_processing(processing_file, protein_id):
            print(f"Features have already been generated for {protein_id}")
            continue

        sequences = individual_json.get('sequences', [])
        contains_ligand = any('ligand' in seq for seq in sequences)
        print(f"Contains ligands: {contains_ligand}")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            if input_json_type == 'server':
                json.dump([individual_json], temp_file)
            else:
                json.dump(individual_json, temp_file)
            temp_file_path = temp_file.name

        add_to_processing(processing_file, protein_id)
        try:
            command = [
                "python", "run_intermediate.py",
                f"--json_path={temp_file_path}",
                f"--model_dir={model_dir}",
                f"--db_dir={db_dir}",
                f"--output_dir={output_dir}",
                "--norun_inference",
                f"--jax_compilation_cache_dir={output_dir}"
            ]
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Feature generation failed for {protein_id}: {e}")
        finally:
            remove_from_processing(processing_file, protein_id)
            os.remove(temp_file_path)

if __name__ == "__main__":
    main()
