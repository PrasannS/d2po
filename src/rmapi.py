from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
# GENERAL CODE FOR RLHF TRAINING ON OUR DIFFERENT SETTINGS

import os
from tqdm import tqdm
from transformers import HfArgumentParser
from trl import set_seed
from contextlib import nullcontext
from torch.utils.data import DataLoader
from datasets import Dataset
from statistics import mean


from utils.rl_utils import (
    load_models, 
)
from utils.args.api_args import APIArguments
from utils.data.dataproc import add_row_index, create_missing_folders_for_file
from utils.data.api_data import trainheuristic_data, reuse_batchdata
from utils.api_utils import get_gold_and_log

from rlhfutils.data import load_manual, tokenize_dset
import torch
from torch import nn

import threading

from rlhfutils.rmcode import RewardDataCollatorWithPadding
import rlhfutils.rewards as rw    

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # for thread safety
    with lock:
        # Get list of strings from the POST request
        data = request.json
        input_texts = data.get("texts", [])

        reward_model.model.config.pad_token_id = reward_model.model.config.eos_token_id
        results = reward_model(input_texts, **sent_kwargs)
        scores = [output[0]["score"] for output in results]
        
        return jsonify(scores)

    
@app.route('/train', methods=['POST'])
def train():
    global metrics
    
    # for thread safety
    with lock:
        metrics['call_count'] += 1
        # Get list of strings from the POST request
        data = request.json
        input_texts = data.get("texts", [])
        # for logging / later use
        metrics['all_texts'].extend(input_texts)
        scores = []
        
        # TODO depending on speed may need to turn off
        for i in range(0, len(input_texts), script_args.batch_size):
            varupdata = (metrics['call_count']%script_args.callratio)==0
            ncont = ((varupdata and (script_args.tracking==False)) and script_args.noupdates==False)
            if ncont: 
                print("WARNING, GRAD UPDATES ON GIVEN DATA")
            # if we want to do the variance updates
            with nullcontext() if ncont else torch.no_grad():
                inps = tokenizer(input_texts[i:i+script_args.batch_size], padding=True, truncation=True, return_tensors="pt").to(reward_model.device)
                rewards = reward_model(input_ids=inps.input_ids, attention_mask=inps.attention_mask)[0]
                scores.extend(rewards.detach().squeeze(-1).tolist())
            
            
            loss = 0 
            print("call count is ", metrics['call_count'])
            
            # Do a batch of validation on normal data
            # if varupdata==False: #(metrics['call_count']%script_args.oldratio)==0:
                
            if varupdata:
                tmploss=0
                if "indiv" in script_args.diffunct:
                    print("doing indiv thing")
                    # TODO update this based on mini-batch size stuff, varmax with oversample
                    for i in range(0, len(rewards), 2):
                        tmploss = tmploss + torch.sigmoid(torch.abs(rewards[i]-rewards[i+1]))
                    # do the variance loss on everything in the batch
                    tmploss = tmploss * (2/len(rewards)) * -1
                elif "batch" in script_args.diffunct:  
                    print("doing dffunct thing")
                    # trick to view all the differences
                    pairwise_diff = rewards.unsqueeze(1) - rewards.unsqueeze(0)
                    # stuff to log for the first time we try this
                    if metrics['call_count']==1:
                        print(pairwise_diff)
                    # use secrets loss
                    tmploss = torch.sigmoid(torch.abs(pairwise_diff)).sum()
                    tmploss = tmploss * (1/(pairwise_diff.shape[1]**2)) * -1
                # NOTE a way to compose the normal update and the new update things
                loss = loss + tmploss
                get_gold_and_log(rewards, inps, input_texts, tokenizer, reward_model, script_args, i, metrics)
                # external method call, note that this is important for preparing data with some methods
            if (metrics['call_count']%script_args.oldratio)==0:
                print("old data update")
                try:
                    inputs = next(loaddata)
                except: 
                    loaddata = iter(train_dataloader)
                    inputs = next(loaddata)
                rewards_j = reward_model(input_ids=inputs["input_ids_j"].to(reward_model.device), attention_mask=inputs["attention_mask_j"].to(reward_model.device))[0]
                rewards_k = reward_model(input_ids=inputs["input_ids_k"].to(reward_model.device), attention_mask=inputs["attention_mask_k"].to(reward_model.device))[0]
                
                rdiff = rewards_j - rewards_k
                
                loss = -nn.functional.logsigmoid(rdiff).mean()

            # we're doing a standard preference update w.r.t the original dataset
            if loss!=0 and script_args.noupdates==False:
                print("replayloss ", float(loss.detach()))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(reward_model.parameters(), max_norm=1.0)
                optimizer.step()
                print(("prompt var step" if "indiv" in script_args.diffunct=='indiv' else "variance step") if varupdata else "retrain step")
        
        # NOTE got rid of weighted value function thing
        metrics['threshsum'].append(mean([abs(scores[i]-scores[i+1]) for i in range(0, len(scores), 2)]))
        metrics['threshsum'] = metrics['threshsum'][-100:] # HACK hardcoded to 100 for now
            
        if script_args.trainheur:
            print("trainheur")
            trainheuristic_data(metrics, tokenizer, script_args, reward_model.device)
        
        if script_args.tracking or script_args.trainheur:
            # take random data points from what we've been messing with
            random.shuffle(metrics['extradata'])
            # newdata = metrics['extradata'][script_args.batch_size*script_args.redo_batches:]
            tmpdset = Dataset.from_list(metrics['extradata'][:script_args.batch_size*script_args.redo_batches]) # HACK this thing is redundant somehow
            # NOTE kicking shuffling back in for stability
            tmpdset = DataLoader(tmpdset, batch_size=script_args.batch_size, shuffle=True, collate_fn=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=script_args.max_length))
            readd_inds = []
            cind = 0
            # do the whole extra adding thing
            for batch in tmpdset:
                
                rewards_j = reward_model(input_ids=batch["input_ids_j"].to(reward_model.device), attention_mask=batch["attention_mask_j"].to(reward_model.device))[0]
                rewards_k = reward_model(input_ids=batch["input_ids_k"].to(reward_model.device), attention_mask=batch["attention_mask_k"].to(reward_model.device))[0]
                
                rdiff = rewards_j - rewards_k # NOTE main bug was here, things should now be fine
                loss = -nn.functional.logsigmoid(rdiff).mean()
                
                ndiff = rdiff.detach()
                
                # TODO make sure that the adding is happening correctly
                reuse_batchdata(ndiff, tokenizer, batch, metrics, readd_inds, script_args, cind, rewards_j, rewards_k)
                cind += script_args.batch_size
                if script_args.noupdates==False:
                    print("activeloss ", float(loss.detach()))
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(reward_model.parameters(), max_norm=1.0)
                    optimizer.step()
            # add back data that needs to be added back
            newdata = list([metrics['extradata'][ind] for ind in readd_inds])
            print("iterated on ", len(metrics['extradata']), " now have ", len(newdata))
            metrics['extradata'] = newdata
                
        print("scores", scores)
            
        if metrics['call_count'] % script_args.save_freq == 0 and (script_args.noupdates==False):
            
            create_missing_folders_for_file(script_args.output_dir+"/step_"+str(metrics['call_count'])+"/")
            reward_model.save_pretrained(script_args.output_dir+"/step_"+str(metrics['call_count']))
            
        return jsonify(scores)
    
if __name__ == '__main__':
    lock = threading.Lock()

    os.environ["WANDB_TAGS"] = "[\"llamatrl\"]"
    tqdm.pandas()
    sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 4, "truncation": True}


    # TODO give this thing its own params
    parser = HfArgumentParser(APIArguments)
    script_args: APIArguments = parser.parse_args_into_dataclasses()[0]

    set_seed(script_args.seed)
    
    if script_args.trainable:
        tokenizer, reward_model = load_models(script_args, "train")
        optimizer = torch.optim.AdamW(reward_model.parameters(), lr=script_args.learning_rate)
        # get the data so that we can update things continually
        train_dataset, evald = load_manual(script_args.dataset_name, "")
        train_dataset = train_dataset.shuffle(seed=100)
        train_dataset = train_dataset.map(add_row_index, with_indices=True)
        evald = evald.map(add_row_index, with_indices=True)
        train_dataset, _ = tokenize_dset(train_dataset, evald, script_args, tokenizer)
        print("updated length is ", len(train_dataset))
        train_dataloader = DataLoader(train_dataset, batch_size=script_args.batch_size, shuffle=True, collate_fn=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=script_args.max_length))
        print("total batches is", len(train_dataloader))
        loaddata = iter(train_dataloader)
        if "contrastivedistill" in script_args.goldreward:
            print("loading the RM")
            rw.likemod = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b", device_map=0, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").eval()
            rw.liketok = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
            rw.slikemod = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", device_map=0, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").eval()
            rw.sliketok = AutoTokenizer.from_pretrained("facebook/opt-125m")
        if script_args.noupdates:
            reward_model.eval()
    else:
        # NOTE handle loading everything in, since hyperparams are same for every setting more or less
        tokenizer, reward_model = load_models(script_args, "rm")

    tokenizer.pad_token = tokenizer.eos_token
    
    metrics = {'call_count':0, 'label_count':0, 'all_texts':[], 'threshsum':[], 'extradata':[], 'logdata':[], 'reuses':{}}

    print("size of tokd thing, ", tokenizer("hi there", padding=True, truncation=True, return_tensors='pt').input_ids.shape)
    
    app.run(debug=False, port=script_args.port)