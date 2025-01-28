'''import json
import subprocess
import tempfile
import fcntl
import os
import shutil

json_file_path = "/storage/coda1/p-jskolnick3/0/jfeldman34/alphafold3/ligand_included.json"
PROCESSING_FILE = "/storage/coda1/p-jskolnick3/0/jfeldman34/alphafold3/processing_file.txt"

def add_to_processing(object_name):
    """
    Add the object name to the shared processing list with file locking.
    """
    with open(PROCESSING_FILE, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)  # Acquire an exclusive lock
        f.seek(0)
        current_objects = f.read().splitlines()
        if object_name not in current_objects:
            f.write(object_name + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)  # Release the lock
        print(f"Processing {object_name}")

def remove_from_processing(object_name):
    """
    Remove the object name from the shared processing list with file locking.
    """
    try:
        with open(PROCESSING_FILE, "r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)  # Acquire an exclusive lock
            current_objects = f.read().splitlines()
            f.seek(0)
            f.truncate()  # Clear the file
            for obj in current_objects:
                if obj != object_name:
                    f.write(obj + "\n")
            fcntl.flock(f, fcntl.LOCK_UN)  # Release the lock
    except FileNotFoundError:
        pass  # If the file doesn't exist, there's nothing to remove

def is_in_processing(object_name):
    """
    Check if an object is already in the shared processing list with file locking.
    """
    try:
        with open(PROCESSING_FILE, "r") as f:
            fcntl.flock(f, fcntl.LOCK_SH)  # Acquire a shared lock
            current_objects = f.read().splitlines()
            fcntl.flock(f, fcntl.LOCK_UN)  # Release the lock
            if(object_name in current_objects):
                print(f"{object_name} already in processing. Skipping...")
            return object_name in current_objects
    except FileNotFoundError:
        return False  # If the file doesn't exist, no objects are being processed

def load_json_objects(file_path):
    """Load a JSON file containing a list of JSON objects and return it as a list."""
    with open(file_path, 'r') as file:
        data = json.load(file)  # Load the list of JSON objects
    return data

# Iterate over each JSON object in the main JSON file
for individual_json in load_json_objects(json_file_path):
    protein_id = individual_json['name']
    output_dir = os.path.join("/storage/coda1/p-jskolnick3/0/jfeldman34/alphafold3/new_output_dir/", individual_json['name'])
    output_lower_dir = os.path.join("/storage/coda1/p-jskolnick3/0/jfeldman34/alphafold3/new_output_dir/", individual_json['name'].lower())
    
    # Check if output already exists
    if os.path.isdir(output_dir) or os.path.isdir(output_lower_dir) or is_in_processing(individual_json['name']):
        print(f"A model has already been generated for {individual_json['name']}")
        continue

    # Check if there are any "ligand" keys in the "sequences" list
    sequences = individual_json.get('sequences', [])
    
    contains_ligand = any('ligand' in seq for seq in sequences)
    print("contains ligand:")
    print(contains_ligand)

    # Create a temporary JSON file for the individual JSON object
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
        json.dump([individual_json], temp_file)
        temp_file_path = temp_file.name

    # Define the command with the temporary file path as the JSON argument
    command = [
        "python", "run_alphafold.py",
        f"--json_path={temp_file_path}",
        "--model_dir=/storage/coda1/p-jskolnick3/0/jfeldman34/scratch/AF3/AF3_Params/",
        "--db_dir=/storage/coda1/p-jskolnick3/0/jfeldman34/scratch/AF3/",
        "--output_dir=/storage/coda1/p-jskolnick3/0/jfeldman34/alphafold3/new_output_dir/"
    ]

    # First subprocess: run the command and check for "ligand" presence
    try:
        add_to_processing(individual_json['name'])
        subprocess.run(command, check=True)
        print(f"First model successfully generated for {individual_json['name']}")

        # If ligand is present, create a modified version of the JSON
        if contains_ligand:
            # Copy the "data.json" file for the protein_id
            protein_folder = os.path.join(output_dir, protein_id.lower())
            data_json_path = os.path.join(protein_folder, "data.json")
            if os.path.exists(data_json_path):
                with open(data_json_path, 'r') as data_file:
                    new_json = json.load(data_file)
                
                # Modify the "name" and remove the sequences with "ligand"
                new_json['name'] = f"{new_json['name']}_without_ligands"
                new_sequences = [seq for seq in new_json.get('sequences', []) if 'ligand' not in seq]
                new_json['sequences'] = new_sequences
                
                # Create a temporary modified JSON file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as new_temp_file:
                    json.dump(new_json, new_temp_file)
                    new_temp_file_path = new_temp_file.name
                    
                print(f"Generating a secondary model for {individual_json['name']}")

                # Run the second subprocess with the modified file
                second_command = [
                    "python", "run_alphafold.py",
                    f"--json_path={new_temp_file_path}",
                    "--model_dir=/storage/coda1/p-jskolnick3/0/jfeldman34/scratch/AF3/AF3_Params/",
                    "--db_dir=/storage/coda1/p-jskolnick3/0/jfeldman34/scratch/AF3/",
                    "--output_dir=/storage/coda1/p-jskolnick3/0/jfeldman34/alphafold3/new_output_dir/"
                ]
                subprocess.run(second_command, check=True)
                print(f"Second model successfully generated for {individual_json['name']}")
                os.remove(new_temp_file_path)
    except subprocess.CalledProcessError as e:
        print(f"Error running AlphaFold script for {individual_json['name']}: {e}")
    finally:
        # Ensure the temporary file is deleted
        os.remove(temp_file_path)
        remove_from_processing(individual_json['name'])

    # After second subprocess, compare ranking_scores
    try:
        protein_folder = os.path.join(output_dir, protein_id.lower())
        protein_summary_path = os.path.join(protein_folder, f"{protein_id.lower()}_data.json_summary_confidences.json")
        without_ligand_folder = os.path.join(output_lower_dir, f"{protein_id.lower()}_without_ligands")
        without_ligand_summary_path = os.path.join(without_ligand_folder, f"{protein_id.lower()}_summary_confidences.json")

        # Extract ranking_score from the protein summary
        with open(protein_summary_path, 'r') as f:
            protein_summary = json.load(f)
        protein_ranking_score = protein_summary.get('ranking_score', -1)

        # Extract ranking_score from the without_ligand summary
        with open(without_ligand_summary_path, 'r') as f:
            without_ligand_summary = json.load(f)
        without_ligand_ranking_score = without_ligand_summary.get('ranking_score', -1)

        # Compare ranking scores and delete the lower folder
        if protein_ranking_score > without_ligand_ranking_score:
            shutil.rmtree(without_ligand_folder)
            os.rename(protein_folder, os.path.join(output_dir, protein_id.lower()))
        else:
            shutil.rmtree(protein_folder)
            os.rename(without_ligand_folder, os.path.join(output_dir, protein_id.lower()))
    except Exception as e:
        print(f"Error comparing ranking scores: {e}")'''
        
import json
import subprocess
import tempfile
import fcntl
import os
import shutil

json_file_path = "/storage/coda1/p-jskolnick3/0/jfeldman34/alphafold3/ligand_included.json"
PROCESSING_FILE = "/storage/coda1/p-jskolnick3/0/jfeldman34/alphafold3/processing_file.txt"

def add_to_processing(object_name):
    """
    Add the object name to the shared processing list with file locking.
    """
    with open(PROCESSING_FILE, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)  # Acquire an exclusive lock
        f.seek(0)
        current_objects = f.read().splitlines()
        if object_name not in current_objects:
            f.write(object_name + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)  # Release the lock
        print(f"Processing {object_name}")

def remove_from_processing(object_name):
    """
    Remove the object name from the shared processing list with file locking.
    """
    try:
        with open(PROCESSING_FILE, "r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)  # Acquire an exclusive lock
            current_objects = f.read().splitlines()
            f.seek(0)
            f.truncate()  # Clear the file
            for obj in current_objects:
                if obj != object_name:
                    f.write(obj + "\n")
            fcntl.flock(f, fcntl.LOCK_UN)  # Release the lock
    except FileNotFoundError:
        pass  # If the file doesn't exist, there's nothing to remove



def is_in_processing(object_name):
    """
    Check if an object is already in the shared processing list with file locking.
    """
    try:
        with open(PROCESSING_FILE, "r") as f:
            fcntl.flock(f, fcntl.LOCK_SH)  # Acquire a shared lock
            current_objects = f.read().splitlines()
            fcntl.flock(f, fcntl.LOCK_UN)  # Release the lock
            if(object_name in current_objects):
                print(f"{object_name} already in processing. Skipping...")
            return object_name in current_objects
    except FileNotFoundError:
        return False  # If the file doesn't exist, no objects are being processed

def load_json_objects(file_path):
    """Load a JSON file containing a list of JSON objects and return it as a list."""
    with open(file_path, 'r') as file:
        data = json.load(file)  # Load the list of JSON objects
    return data

# Iterate over each JSON object in the main JSON file
for individual_json in load_json_objects(json_file_path):
    protein_id = individual_json['name']
    output_dir = os.path.join("/storage/coda1/p-jskolnick3/0/jfeldman34/alphafold3/new_output_dir/")
    output_lower_dir = os.path.join("/storage/coda1/p-jskolnick3/0/jfeldman34/alphafold3/new_output_dir/")
    output_dir_check = os.path.join("/storage/coda1/p-jskolnick3/0/jfeldman34/alphafold3/new_output_dir/", individual_json['name'])
    output_lower_dir_check = os.path.join("/storage/coda1/p-jskolnick3/0/jfeldman34/alphafold3/new_output_dir/", individual_json['name'].lower())
        
        
    # Check if output already exists
    if os.path.isdir(output_dir_check) or os.path.isdir(output_lower_dir_check) or is_in_processing(individual_json['name']):
        print(f"A model has already been generated for {individual_json['name']}")
        continue
    
    # Check if there are any "ligand" keys in the "sequences" list
    sequences = individual_json.get('sequences', [])
    
    contains_ligand = any('ligand' in seq for seq in sequences)
    print(f"Contains ligands: contains_ligand")
    
    # Create a temporary JSON file for the individual JSON object
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
        json.dump([individual_json], temp_file)
        temp_file_path = temp_file.name
    
    # Define the command with the temporary file path as the JSON argument
    command = [
        "python", "run_alphafold.py",
        f"--json_path={temp_file_path}",
        "--model_dir=/storage/coda1/p-jskolnick3/0/jfeldman34/scratch/AF3/AF3_Params/",
        "--db_dir=/storage/coda1/p-jskolnick3/0/jfeldman34/scratch/AF3/",
        "--output_dir=/storage/coda1/p-jskolnick3/0/jfeldman34/alphafold3/new_output_dir/"
    ]
    
    # First subprocess: run the command and check for "ligand" presence
    try:
        add_to_processing(individual_json['name'])
        subprocess.run(command, check=True)
        print(f"First model successfully generated for {individual_json['name']}")
        
    
        # If ligand is present, create a modified version of the JSON
        if contains_ligand:
            # Copy the "data.json" file for the protein_id
            protein_folder = os.path.join(output_dir, protein_id.lower())
            print(protein_folder)
            data_json_path = os.path.join(protein_folder, f"{protein_id.lower()}_data.json")
            print(data_json_path)
            if os.path.exists(data_json_path):
                with open(data_json_path, 'r') as data_file:
                    new_json = json.load(data_file)
                
                # Modify the "name" and remove the sequences with "ligand"
                new_json['name'] = f"{new_json['name']}_without_ligands"
                new_sequences = [seq for seq in new_json.get('sequences', []) if 'ligand' not in seq]
                new_json['sequences'] = new_sequences
                
                # Create a temporary modified JSON file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as new_temp_file:
                    json.dump(new_json, new_temp_file) 
                    new_temp_file_path = new_temp_file.name
                    print(new_temp_file_path)
                    
                print(f"Generating a secondary model for {individual_json['name']}")
    
                # Run the second subprocess with the modified file
                second_command = [
                    "python", "run_alphafold.py",
                    f"--json_path={new_temp_file_path}",
                    "--model_dir=/storage/coda1/p-jskolnick3/0/jfeldman34/scratch/AF3/AF3_Params/",
                    "--db_dir=/storage/coda1/p-jskolnick3/0/jfeldman34/scratch/AF3/",
                    "--output_dir=/storage/coda1/p-jskolnick3/0/jfeldman34/alphafold3/new_output_dir/"
                ]
                subprocess.run(second_command, check=True)
                print(f"Second model successfully generated for {individual_json['name']}")
                os.remove(new_temp_file_path)
    except subprocess.CalledProcessError as e:
        print(f"Error running AlphaFold script for {individual_json['name']}: {e}")
    finally:
        # Ensure the temporary file is deleted
        os.remove(temp_file_path)
        remove_from_processing(individual_json['name'])
    
    # After second subprocess, compare ranking_scores
    try:
        protein_folder = os.path.join(output_dir, protein_id.lower())
        protein_summary_path = os.path.join(protein_folder, f"{protein_id.lower()}_summary_confidences.json")
        without_ligand_folder = os.path.join(output_lower_dir, f"{protein_id.lower()}_without_ligands")
        without_ligand_summary_path = os.path.join(without_ligand_folder, f"{protein_id.lower()}_without_ligands_summary_confidences.json")
    
        # Extract ranking_score from the protein summary
        with open(protein_summary_path, 'r') as f:
            protein_summary = json.load(f)
        protein_ranking_score = protein_summary.get('ranking_score', -1)
    
        # Extract ranking_score from the without_ligand summary
        with open(without_ligand_summary_path, 'r') as f:
            without_ligand_summary = json.load(f)
        without_ligand_ranking_score = without_ligand_summary.get('ranking_score', -1)
    
        # Compare ranking scores and delete the lower folder
        if protein_ranking_score > without_ligand_ranking_score:
            shutil.rmtree(without_ligand_folder)
            os.rename(protein_folder, os.path.join(output_dir, protein_id.lower()))
        else:
            shutil.rmtree(protein_folder)
            os.rename(without_ligand_folder, os.path.join(output_dir, protein_id.lower()))
    except Exception as e:
        print(f"Error comparing ranking scores: {e}")

