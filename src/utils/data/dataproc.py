# THIS IS A FILE CONTAINING GENERIC CODE RELATED TO PROCESSING / LOADING / SAVING DATA

import os 
import json 
import filelock 
        
def add_row_index(example, idx):
    example['row_index'] = idx
    return example

def create_missing_folders_for_file(path):
    # Extract directory path from the file path
    directory = os.path.dirname(path)
    lock = filelock.FileLock(f".tmp.lock")
    with lock:
        # Check if the directory already exists
        if not os.path.exists(directory):
            # If it does not exist, create it including any necessary parent directories
            os.makedirs(directory)
            print(f"NOTE: Just created missing folders for: {path}")
        
def append_dict_to_jsonl(metadata_dict, fname, lis=None):
    
    # Convert dictionary to JSON string
    json_string = json.dumps(metadata_dict)
    create_missing_folders_for_file(fname)
    # Ensure thread-safe file writing
    lock = filelock.FileLock(f"{fname}.lock")

    with lock:
        
        # Open the file in append mode and write the JSON string
        with open(fname, 'a') as file:
            file.write(json_string + '\n')
        if lis!=None:
            lis.append(metadata_dict)