import json
import subprocess
import tempfile
import fcntl
import os
import shutil
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run AF3Complex Inference. You must have already generated your features!")
    parser.add_argument("--feature_dir_path", required=True, help="Path to the directory containing protein feature folders.")
    parser.add_argument("--model_dir", required=True, help="Path to the model directory.")
    parser.add_argument("--db_dir", required=True, help="Path to the database directory.")
    parser.add_argument("--output_dir", required=True, help="Path to the model output directory.")
    return parser.parse_args()

def get_processing_file_path(feature_dir_path):
    return os.path.join(feature_dir_path, "inference_processing_file.txt")

def get_lock_file_path(processing_file):
    return processing_file + ".lock"

def try_claim_protein_for_processing(processing_file, protein_id, output_dir):
    
    lock_file_path = get_lock_file_path(processing_file)
    
  
    os.makedirs(os.path.dirname(lock_file_path), exist_ok=True)
    
    
    with open(lock_file_path, "a"):
        pass
    
    with open(lock_file_path, "r") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        
        try:
            
            output_dir_check = os.path.join(output_dir, protein_id)
            output_lower_dir_check = os.path.join(output_dir, protein_id.lower())
            
            if os.path.isdir(output_dir_check) or os.path.isdir(output_lower_dir_check):
                print(f"Output directory already exists for {protein_id}")
                return False
                
            
            if os.path.exists(processing_file):
                with open(processing_file, "r") as f:
                    current_objects = f.read().splitlines()
                if protein_id in current_objects:
                    print(f"{protein_id} already in processing. Skipping...")
                    return False
            

            if not os.path.exists(processing_file):
                open(processing_file, 'w').close()
                
            with open(processing_file, "a") as f:
                f.write(protein_id + "\n")
            
            print(f"Successfully claimed {protein_id} for processing")
            return True
            
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)

def remove_from_processing(processing_file, protein_id):
    lock_file_path = get_lock_file_path(processing_file)
    
    try:
        if not os.path.exists(lock_file_path):
            return
            
        with open(lock_file_path, "r") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            
            try:
                if not os.path.exists(processing_file):
                    return
                    
                with open(processing_file, "r") as f:
                    current_objects = f.read().splitlines()
                
                with open(processing_file, "w") as f:
                    for obj in current_objects:
                        if obj != protein_id:
                            f.write(obj + "\n")
                            
                print(f"Removed {protein_id} from processing")
                
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
                
    except FileNotFoundError:
        pass

def main():
    args = parse_arguments()
    feature_dir_path = args.feature_dir_path
    model_dir = args.model_dir
    db_dir = args.db_dir
    output_dir = args.output_dir
    processing_file = get_processing_file_path(feature_dir_path)

    for subdir in os.listdir(feature_dir_path):
        subfolder_path = os.path.join(feature_dir_path, subdir)
        if not os.path.isdir(subfolder_path):
            continue

        json_files = [f for f in os.listdir(subfolder_path) if f.endswith("data.json")]
        if not json_files:
            print(f"No data.json file found in {subdir}, skipping.")
            continue

        json_path = os.path.join(subfolder_path, json_files[0])
        
        with open(json_path, 'r') as f:
            json_obj = json.load(f)

        protein_id = json_obj['name']
        
        if not try_claim_protein_for_processing(processing_file, protein_id, output_dir):
            print(f"A model has already been generated for {protein_id}")
            continue

        sequences = json_obj.get('sequences', [])
        contains_ligand = any('ligand' in seq for seq in sequences)
        print(f"Contains ligands: {contains_ligand}")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump(json_obj, temp_file)
            temp_file_path = temp_file.name

        command = [
            "python", "run_intermediate.py",
            f"--json_path={temp_file_path}",
            f"--model_dir={model_dir}",
            f"--db_dir={db_dir}",
            f"--output_dir={output_dir}",
            "--norun_data_pipeline",
            f"--jax_compilation_cache_dir={feature_dir_path}"
        ]

        new_temp_file_path = None

        try:
            subprocess.run(command, check=True)
            print(f"First model successfully generated for {protein_id}")

            if contains_ligand:
                protein_folder = os.path.join(output_dir, protein_id.lower())
                data_json_path = os.path.join(protein_folder, f"{protein_id.lower()}_data.json")
                if os.path.exists(data_json_path):
                    with open(data_json_path, 'r') as data_file:
                        new_json = json.load(data_file)

                    new_json['name'] = f"{new_json['name']}_without_ligands"
                    new_json['sequences'] = [seq for seq in new_json.get('sequences', []) if 'ligand' not in seq]

                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as new_temp_file:
                        json.dump(new_json, new_temp_file)
                        new_temp_file_path = new_temp_file.name

                    print(f"Generating a secondary model for {protein_id}")

                    second_command = [
                        "python", "run_intermediate.py",
                        f"--json_path={new_temp_file_path}",
                        f"--model_dir={model_dir}",
                        f"--db_dir={db_dir}",
                        f"--output_dir={output_dir}",
                        f"--norun_data_pipeline",
                        f"--jax_compilation_cache_dir={feature_dir_path}"
                    ]

                    subprocess.run(second_command, check=True)
                    print(f"Second model successfully generated for {protein_id}")

        except subprocess.CalledProcessError as e:
            print(f"Error running AlphaFold script for {protein_id}: {e}")

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            remove_from_processing(processing_file, protein_id)
            if new_temp_file_path and os.path.exists(new_temp_file_path):
                os.remove(new_temp_file_path)

        if contains_ligand:
            try:
                protein_folder = os.path.join(output_dir, protein_id.lower())
                protein_summary_path = os.path.join(protein_folder, f"{protein_id.lower()}_summary_confidences.json")
                without_ligand_folder = os.path.join(output_dir, f"{protein_id.lower()}_without_ligands")
                without_ligand_summary_path = os.path.join(without_ligand_folder, f"{protein_id.lower()}_without_ligands_summary_confidences.json")

                with open(protein_summary_path, 'r') as f:
                    protein_summary = json.load(f)
                with open(without_ligand_summary_path, 'r') as f:
                    without_ligand_summary = json.load(f)

                score = protein_summary.get('ranking_score', -1)
                score_wo = without_ligand_summary.get('ranking_score', -1)

                print(f"With ligands score: {score}")
                print(f"Without ligands score: {score_wo}")

                if score > score_wo:
                    shutil.rmtree(without_ligand_folder)
                else:
                    shutil.rmtree(protein_folder)
                    os.rename(without_ligand_folder, protein_folder)
            except Exception as e:
                print(f"Error comparing ranking scores: {e}")

if __name__ == "__main__":
    main()
