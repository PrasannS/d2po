from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
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
import numpy as np

from utils.rl_utils import (
    load_models, 
)
from utils.args.api_args import APIArguments
from utils.data.dataproc import add_row_index, create_missing_folders_for_file
from utils.data.api_data import reuse_batchdata
from utils.api_utils import get_gold_and_log
from utils.misc.loss_utils import get_batch_logps
from utils.data.prompt_utils import convert_prompstlye, apfarmstyle, cat

from rlhfutils.data import load_manual, tokenize_dset
import torch
from torch import nn

import threading

from rlhfutils.rmcode import RewardDataCollatorWithPadding
import utils.eval.rewards as rw

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
    
@app.route('/predeurus', methods=['POST'])
def predeurus():
    # for thread safety
    with lock:
        # Get list of strings from the POST request
        data = request.json
        input_texts = data.get("texts", [])
        
        return rw.eurus_rm(input_texts)

    
# get DPO rewards with ref, reward model
def dpo_infer(dpoinputs):
    with torch.no_grad():
        rmlogs = reward_model(input_ids=dpoinputs.input_ids, attention_mask=dpoinputs.attention_mask).logits
        reflogs = ref_model(input_ids=dpoinputs.input_ids, attention_mask=dpoinputs.attention_mask).logits
        rmlps = get_batch_logps(rmlogs, dpoinputs.input_ids, label_pad_token_id=tokenizer.pad_token_id)
        reflps = get_batch_logps(reflogs, dpoinputs.input_ids, label_pad_token_id=tokenizer.pad_token_id)
    return rmlps - reflps

BETA=0.05
relabelamt = 0
rmready=True
@app.route('/train', methods=['POST'])
def train():
    global metrics, relabelamt, rmready
    
    # for thread safety, TODO sanity check, length-based threshold should work
    with lock:
        redodata = []
        # just had a reset before this
        if len(metrics['all_texts'])==0:
            relabelamt=script_args.relabels
        metrics['call_count'] += 1
        # Get list of strings from the POST request
        data = request.json
        input_texts = data.get("texts", [])
        
        if script_args.usedpo:
            # switch up prompt style, TODO make more flexible later
            if script_args.goldreward=='contrastivedistill':
                input_texts = [convert_prompstlye(inpt, cat) for inpt in input_texts]
            else:
                input_texts = [convert_prompstlye(inpt, apfarmstyle) for inpt in input_texts]
                
        script_args.inpsize = len(input_texts)
        # for logging / later use
        metrics['all_texts'].extend(input_texts)
        metrics['cinds'].extend([metrics['call_count']]*len(input_texts))
        
        print("call count is ", metrics['call_count'])
        
        # compute scores on current batch, get all scores, inps, attentions
        # TODO depending on speed may need to turn off
        for i in range(0, len(input_texts), script_args.batch_size):
            # if we want to do the variance updates
            with torch.no_grad():
                inps = tokenizer(input_texts[i:i+script_args.batch_size], padding=True, truncation=True, return_tensors="pt").to(reward_model.device)
                if script_args.usedpo:
                    rewards = dpo_infer(inps)
                else:
                    rewards = reward_model(input_ids=inps.input_ids, attention_mask=inps.attention_mask)[0]
                # keep a working list of things we pass in here, they're going to get used later in training
                metrics['inpids'].extend(inps.input_ids.detach())
                metrics['masks'].extend(inps.attention_mask.detach())
                metrics['rscores'].extend(rewards.detach().squeeze(-1).tolist())
        
        returnscos = metrics['rscores'][-len(input_texts):]
        
        assert len(metrics['masks'])==len(metrics['rscores'])
        assert len(metrics['masks'])==len(metrics['all_texts'])
        
        print("ready to do some RM updates")
        # only take stuff from the last bit
        inds = list(range(len(metrics['rscores']) - len(input_texts), len(metrics['rscores']), 2))
        # can use either random-based or slightly more complex confidence thing for selecting examples to use in active updates
        # NOTE this will change how selection works a bit
        if "conf" in script_args.relab_criteria:
            diffs = list([abs(float(metrics['rscores'][i]-metrics['rscores'][i+1])) for i in range(len(metrics['rscores']) - len(input_texts), len(metrics['rscores']), 2)])
            inds = list([inds[i] for i in np.argsort(diffs)])
        elif "rand" in script_args.relab_criteria:
            random.shuffle(inds)
        
        # we need to make sure we're not over-using gold data in this format (TODO I guess simplest for now is to just draw from first chance batch?)
        inds = inds[:relabelamt]
        if relabelamt>0:
            olen = len(inds)
            if relabelamt>olen:
                relabelamt = relabelamt-olen
            else:
                relabelamt=0
        
            
            
        print("we're going to now train with: ", len(inds), " preference pairs out of a possible ", len(metrics['rscores'])/2)
        
        # once we've figured out that we want to update things, let's now select something with a strategy, get golds, etc. 
        returnscos = get_gold_and_log(inds, tokenizer, script_args, metrics)
        returnscos = list([float(r) for r in returnscos])
        # TODO need to make sure gold score correction is happening correctly
        
        print("we have: ", len(metrics['extradata']), "new examples to work with when throwing out equal pairs")
        
        # break apart collection / training on data from clearing
        # we're ready to do some kind of active update
        if rmready and relabelamt==0:
            rmready=False
            if script_args.noupdates or (len(metrics['extradata'])==0): 
                # reset everything now
                resetkeys = ["extradata", 'inpids', 'masks', 'rscores', 'all_texts', 'cinds']
                for r in resetkeys:
                    metrics[r] = []
                    metrics[r].clear()
                print("early return")
                assert len(metrics['masks'])==0
                return jsonify(returnscos)
            # take random data points from what we've been messing with
            # random.shuffle(metrics['extradata'])
            # NOTE added redodata here
            tmpdset = Dataset.from_list(metrics['extradata']+redodata)
            print("len of new data is ", len(metrics['extradata']))
            
            if script_args.oldupdates:
                print("use old data instead")
                tmpdset = DataLoader(train_dataset.select(range(metrics['data_pointer'], metrics['data_pointer']+len(tmpdset))), batch_size=script_args.batch_size, shuffle=True, collate_fn=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=script_args.max_length))
                metrics['data_pointer'] += len(tmpdset)
            else:
                # NOTE kicking shuffling back in for stability
                tmpdset = DataLoader(tmpdset, batch_size=script_args.batch_size, shuffle=True, collate_fn=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=script_args.max_length))
                
            # cind = 0 readd_inds = []
            for e in range(script_args.update_epochs):
                
                # do the whole extra adding thing
                for idx, batch in enumerate(tmpdset):
                    
                    # TODO is there some way to move this out of this method?
                    # using DPO loss (assume we're working w a causal LM)
                    if script_args.usedpo:
                        # get logprobs, then we can do DPO loss update
                        logits_j = reward_model(input_ids=batch["input_ids_j"].to(reward_model.device), attention_mask=batch["attention_mask_j"].to(reward_model.device)).logits
                        logits_k = reward_model(input_ids=batch["input_ids_k"].to(reward_model.device), attention_mask=batch["attention_mask_k"].to(reward_model.device)).logits
                        lps_j = get_batch_logps(logits_j, batch["input_ids_j"].to(reward_model.device), label_pad_token_id=tokenizer.pad_token_id)
                        lps_k = get_batch_logps(logits_k, batch["input_ids_k"].to(reward_model.device), label_pad_token_id=tokenizer.pad_token_id)
                        # get the same thing but for the reference model
                        with torch.no_grad():
                            rlogits_j = ref_model(input_ids=batch["input_ids_j"].to(reward_model.device), attention_mask=batch["attention_mask_j"].to(reward_model.device)).logits
                            rlogits_k = ref_model(input_ids=batch["input_ids_k"].to(reward_model.device), attention_mask=batch["attention_mask_k"].to(reward_model.device)).logits
                            rlps_j = get_batch_logps(rlogits_j, batch["input_ids_j"].to(reward_model.device), label_pad_token_id=tokenizer.pad_token_id)
                            rlps_k = get_batch_logps(rlogits_k, batch["input_ids_k"].to(reward_model.device), label_pad_token_id=tokenizer.pad_token_id)
                        pi_logratios = lps_j - lps_k
                        ref_logratios = rlps_j - rlps_k
                        
                        pi_logratios = pi_logratios.to(ref_model.device)
                        ref_logratios = ref_logratios.to(ref_model.device)
                        logits = pi_logratios - ref_logratios
                        loss = -nn.functional.logsigmoid(BETA * logits).mean()
                        
                    else:
                        # normal RM BT loss
                        rewards_j = reward_model(input_ids=batch["input_ids_j"].to(reward_model.device), attention_mask=batch["attention_mask_j"].to(reward_model.device))[0]
                        rewards_k = reward_model(input_ids=batch["input_ids_k"].to(reward_model.device), attention_mask=batch["attention_mask_k"].to(reward_model.device))[0]
                        
                        rdiff = rewards_j - rewards_k # NOTE main bug was here, things should now be fine
                        loss = -nn.functional.logsigmoid(rdiff).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(reward_model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    # ndiff = rdiff.detach() NOTE leaving this here in case we want XP replay back
                    # reuse_batchdata(ndiff, tokenizer, batch, metrics, readd_inds, script_args, cind, rewards_j, rewards_k)
                    # cind += script_args.batch_size
                    
                    print(idx, ": epoch ", e, ": activeloss ", float(loss.detach()))
                    
            # reset everything now
            resetkeys = ["extradata", 'inpids', 'masks', 'rscores', 'all_texts', 'cinds']
            if script_args.redo_batches>0:
                redodata.extend(metrics['extradata'])
                # take certain amount of data to re-use
                redodata = redodata[-script_args.batch_size*script_args.redo_batches:]
            
            # add back data that needs to be added back
            # newdata = list([metrics['extradata'][ind] for ind in readd_inds])
            # print("iterated on ", len(metrics['extradata']), " now have ", len(newdata))
            # metrics['extradata'] = newdata
            
            metrics['label_count']+=len(inds)
            if metrics['label_count']>=script_args.stopupdates:
                print("DISABLING RM UPDATES FROM THIS POINT")
                script_args.noupdates=True
                reward_model.eval()
        # only clear / move on once we've hit samp_n steps
        if script_args.samp_n<=len(metrics['masks']):
            resetkeys = ["extradata", 'inpids', 'masks', 'rscores', 'all_texts', 'cinds']
            rmready=True
            for r in resetkeys:
                metrics[r] = []
                metrics[r].clear()

            assert len(metrics['masks'])==0
                
        if metrics['call_count'] % script_args.save_freq == 0 and (script_args.noupdates==False):
            create_missing_folders_for_file(script_args.output_dir+"/step_"+str(metrics['call_count'])+"/")
            reward_model.save_pretrained(script_args.output_dir+"/step_"+str(metrics['call_count']))
        
            
        return jsonify(returnscos)
    
if __name__ == '__main__':
    lock = threading.Lock()

    os.environ["WANDB_TAGS"] = "[\"llamatrl\"]"
    tqdm.pandas()
    sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 4, "truncation": True}


    # TODO give this thing its own params
    parser = HfArgumentParser(APIArguments)
    script_args: APIArguments = parser.parse_args_into_dataclasses()[0]

    set_seed(script_args.seed)        
    
    metrics = {'call_count':0, 'label_count':0, 'all_texts':[], 'all_scores':[], 'inpids':[], 'masks':[], 'rscores':[], 'cinds':[], 'extradata':[], 'logdata':[], 'reuses':{}, 'data_pointer':0}

    print("goldreward is ", script_args.goldreward)
    if "eurusrm_fresh" in script_args.reward_model_name:
        print("fresh eurus thing")
        rw.sliketok = AutoTokenizer.from_pretrained("openbmb/Eurus-RM-7b")
        rw.slikemod = AutoModel.from_pretrained("openbmb/Eurus-RM-7b", device_map=0, torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
        
    elif script_args.trainable:
        tokenizer, reward_model = load_models(script_args, "train")
        optimizer = torch.optim.AdamW(reward_model.parameters(), lr=script_args.learning_rate)
        # get the data so that we can update things continually
        if len(script_args.dataset_name)>0 and script_args.dataset_name[-1]!="/":
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
            metrics['contdist'] = [
                AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b", device_map=0, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").eval(),
                AutoTokenizer.from_pretrained("facebook/opt-1.3b"),
                AutoModelForCausalLM.from_pretrained("facebook/opt-125m", device_map=0, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").eval(),
                AutoTokenizer.from_pretrained("facebook/opt-125m")
            ]        
        if "eurusrm" in script_args.goldreward:
            # RM with its own custom code setup
            metrics['contdist'] = [
                AutoModel.from_pretrained("openbmb/Eurus-RM-7b", device_map=0, torch_dtype=torch.bfloat16, trust_remote_code=True).eval(),
                AutoTokenizer.from_pretrained("openbmb/Eurus-RM-7b"),
            ]     
            
        if script_args.usedpo:
            ref_model = AutoModelForCausalLM.from_pretrained(script_args.dpobase, device_map=0, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").eval()
            
        if script_args.noupdates:
            reward_model.eval()
        tokenizer.pad_token = tokenizer.eos_token
        print("size of tokd thing, ", tokenizer("hi there", padding=True, truncation=True, return_tensors='pt').input_ids.shape)
    else:
        # NOTE handle loading everything in, since hyperparams are same for every setting more or less
        tokenizer, reward_model = load_models(script_args, "rm")

        tokenizer.pad_token = tokenizer.eos_token
    

        print("size of tokd thing, ", tokenizer("hi there", padding=True, truncation=True, return_tensors='pt').input_ids.shape)
    
    app.run(debug=False, port=script_args.port)