# THIS IS A FILE CONTAINING GENERIC CODE RELATED TO PROCESSING / LOADING / SAVING DATA

import os 
import json 
import filelock 
from datasets import Dataset
from utils.eval.rewards import annot_proc

        
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

# given a starting index (default 0), and some other thing, return the
# better and worse index in that order
def indcomp(rws, sind=0, eind=-1): 
    # extra functionalty
    if eind>=0: 
        return (sind, eind) if rws[sind]>rws[eind] else (eind, sind)
    if rws[0]>rws[1]:
        return 0+sind, 1+sind
    return 1+sind, 0+sind

# given a dataframe with a "golds" column, make rollouts into preference data
# (assume that pairs go together as rollouts)
def annot_to_prefs(indf, dropreps=True):
    prefdata = []
    for ind, row in indf.iterrows():
        inp = row['instruction']
        resps = [row['completions'][i]['response'] for i in range(len(row['completions']))]
        golds = annot_proc(row)
        for i in range(len(golds)): 
            for j in range(i+1, len(golds)): 
                if i==j:
                    continue
                indj, indk = indcomp(golds, i, j)
                tmp = {
                    'question': inp,
                    'response_j': resps[indj],
                    'response_k': resps[indk],
                    'score_j': golds[indj], 
                    'score_k': golds[indk],
                    'magnitude': golds[indj] - golds[indk]
                }
                prefdata.append(tmp)
    newdata =  Dataset.from_list(prefdata)
    return newdata.filter(lambda ex: ex['magnitude']>0) if dropreps else newdata
    
# given a dataframe with a "golds" column, make rollouts into preference data
# (assume that pairs go together as rollouts)
def rollout_to_prefs(indf):
    prefdata = []
    for ind, row in indf.iterrows():
        for i in range(0, len(row['inputs']), 2):
            assert row['inputs'][i]==row['inputs'][i+1]
            indj, indk = indcomp(row['golds'][i:i+2], i)
            tmp = {
                'question': row['inputs'][i],
                'response_j': row['outputs'][indj],
                'response_k': row['outputs'][indk],
                'score_j': row['golds'][indj], 
                'score_k': row['golds'][indk],
                'magnitude': row['golds'][indj] - row['golds'][indk]
            }
            prefdata.append(tmp)
            
    return Dataset.from_list(prefdata)