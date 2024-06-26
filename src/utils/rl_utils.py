# All the verbose / complex stuff that's generic to all TRL RLHF code
import torch

from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import (
    Adafactor,
    AutoTokenizer,
    LlamaTokenizer,
    pipeline,
)
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoModel
from peft import TaskType
import numpy as np
from nltk.tokenize import word_tokenize
import random
from statistics import mean
import json
import math
from utils.misc.loss_utils import get_batch_logps
# HACK setting up forward in alternative fashion
# # NOTE bring omit_long back
from utils.eval.rewards import get_synth_rewards, omit_long
import utils.eval.rewards as rw
from utils.data.dataproc import append_dict_to_jsonl, create_missing_folders_for_file
from utils.misc.logit_procs import EinsteinLogitsProc

import requests, json

from statistics import mean, stdev
import random

from trl import AutoModelForCausalLMWithValueHead, PPOConfig
from trl.core import LengthSampler

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

def get_pipeline(rmname, device):
    pipetok = AutoTokenizer.from_pretrained(rmname)
    # HACK this used to be in if statement, note that RM tokenizer code always sets pad to eos, making things weird
    # if pipetok.pad_token is None:
    pipetok.pad_token_id = pipetok.eos_token_id
    pipetok.pad_token = pipetok.eos_token
    print("using pipeline with pad token ", pipetok.pad_token)
    reward_model = pipeline(
        "sentiment-analysis",
        model=rmname,
        device_map={"": device},
        model_kwargs={"torch_dtype": torch.bfloat16},
        tokenizer=pipetok,
        return_token_type_ids=False,
    )
    reward_model.tokenizer.pad_token = reward_model.tokenizer.eos_token
    reward_model.model.config.pad_token_id = reward_model.tokenizer.eos_token_id
    
    # sanity check
    print("pipeline sanity ", reward_model(['hi there', 'my friend']))
    
    return pipetok, reward_model

def load_models(script_args, loadms="rmppo", dev=0):
    
    current_device = Accelerator().local_process_index

    # TODO do I somehow need this for more stuff?
    if "decapoda" in script_args.model_name.lower():
        tokenizer = LlamaTokenizer.from_pretrained(script_args.model_name)
        # required for llama
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
                "pad_token": DEFAULT_PAD_TOKEN,
            }
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
        if getattr(tokenizer, "pad_token", None) is None:
            print("resetting pad token?")
            tokenizer.pad_token = tokenizer.eos_token
            # tokenizer.pad_token_type_id = tokenizer.eos_token_id
    
    try:
        rname = script_args.output_dir.split("/")
        rname = rname[-1] if rname[-1]!="" else rname[-2]
    except:
        rname = "nothing"
    if "ppo" in loadms:
        
        config = PPOConfig(
            model_name=script_args.model_name,
            learning_rate=script_args.learning_rate,
            log_with="wandb",
            batch_size=script_args.batch_size,
            mini_batch_size=script_args.mini_batch_size,
            gradient_accumulation_steps=script_args.gradient_accumulation_steps,
            optimize_device_cache="dpoplus" not in script_args.kl_penalty,
            early_stopping=script_args.early_stopping,
            target_kl=script_args.target_kl,
            ppo_epochs=script_args.ppo_epochs,
            seed=script_args.seed,
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=.1,
            horizon=10000,
            target=script_args.target_kl,
            init_kl_coef=script_args.init_kl_coef,
            steps=script_args.steps,
            gamma=1,
            lam=0.95,
            kl_penalty=script_args.kl_penalty, 
            remove_unused_columns=False,
            tracker_project_name=rname, 
        )
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            script_args.model_name,
            load_in_8bit=True, # re-enable for llama model
            device_map={"": current_device},
            peft_config=lora_config, 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        if script_args.self_reward_steps > 0: 
            
            refmodel = AutoModelForCausalLMWithValueHead.from_pretrained(
                "facebook/opt-125m",
                load_in_8bit=True, # re-enable for llama model
                device_map={"": current_device},
                peft_config=lora_config, 
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
            refmodel.eval()

        optimizer = None
        if script_args.adafactor:
            optimizer = Adafactor(
                filter(lambda p: p.requires_grad, model.parameters()),
                scale_parameter=False,
                relative_step=False,
                warmup_init=False,
                lr=config.learning_rate,
            )

    if "train" in loadms:
        
        # we want to train the RM on the fly
        tokenizer_name = script_args.tokenizer_name if script_args.tokenizer_name is not None else script_args.reward_model_name
        print("toker name is", tokenizer_name)
        print(script_args)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_auth_token=True)
        tokenizer.pad_token = tokenizer.eos_token
    
        if script_args.usedpo: 
            print("using a dpo based RM")
        else:
            print("using a normal RM")
        # keep same for DPO, RM (note )
        peft_config = LoraConfig(
            task_type= TaskType.CAUSAL_LM if script_args.usedpo else TaskType.SEQ_CLS ,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        
        modtype = AutoModelForCausalLM if script_args.usedpo else AutoModelForSequenceClassification
        model = modtype.from_pretrained(
            script_args.reward_model_name, num_labels=1, torch_dtype=torch.bfloat16, device_map=dev # device_map="auto"
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        # Need to do this for gpt2, because it doesn't have an official pad token.
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
        model.config.use_cache = True
        return tokenizer, model
    
    if "rm" in loadms:
        ptok, reward_model = get_pipeline(script_args.reward_model_name, current_device)
    if loadms=="rm":
        return  tokenizer, reward_model
    model.gradient_checkpointing_disable()
    # PPO client for API endpoint
    if loadms=="ppo":
        # standard PPO
        if script_args.self_reward_steps>0:
            return config, tokenizer, (model, refmodel), optimizer
        return config, tokenizer, model, optimizer
    
    return config, tokenizer, model, optimizer, reward_model

def lensco(lval):
    return -1*abs(lval-1)+1

def get_scores_from_api(text_list, url):
    # URL of your Flask API
    # url = 'http://127.0.0.1:5000/predict'

    # Prepare data in the correct format
    data = json.dumps({"texts": text_list})

    # Send POST request to the Flask API
    try:
        response = requests.post(url, data=data, headers={'Content-Type': 'application/json'}, timeout=1000)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Get the scores from the response
        scores = response.json()
        return scores
    except requests.exceptions.HTTPError as errh:
        print(f"Http Error: {errh}")
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting: {errc}")
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        print(f"Oops: Something Else: {err}")
        

    
def keep_strat(script_args, rewards, keep_inds):
    keeplen = int(len(rewards)/script_args.oversample)
    # this will get us a "maximum pair" to do DPO with, if we wanted random that would be equiv to oversample 1
    # preferred first then dispreferred
    dpoplus = "dpoplus" in script_args.kl_penalty 
    # only keep output of each prompt that is the best or worst in terms of reward 
    if ('prompt_max' in script_args.rollout_strategy) or dpoplus:
        keep_inds = []
        # within each prompt sample
        for i in range(0, len(rewards), script_args.oversample):
            keep_inds.append(int(i+np.argmax(rewards[i:i+script_args.oversample])))
    if ('prompt_min' in script_args.rollout_strategy) or dpoplus:
        if dpoplus is False: 
            keep_inds = []
        # within each prompt sample
        for i in range(0, len(rewards), script_args.oversample):
            keep_inds.append(int(i+np.argmin(rewards[i:i+script_args.oversample])))
    # 'all': take regardless of where it's coming from
    elif 'all_max' in script_args.rollout_strategy:
        # last of set to keee
        keep_inds = list(np.argsort(rewards))[-(keeplen):]
        assert rewards[keep_inds[-1]]==max(rewards)
    elif 'all_min' in script_args.rollout_strategy:
        keep_inds = list(np.argsort(rewards))[:(keeplen)]
        assert rewards[keep_inds[0]]==min(rewards)
    if script_args.oversample>1:
        varlist = [stdev(rewards[i:i+script_args.oversample]) for i in range(0, len(rewards), script_args.oversample)]
    # keep all stuff from prompts with the highest variation
    if 'var_max' in script_args.rollout_strategy:
        keep_inds = []
        keepvars = list(np.argsort(varlist))[-int(keeplen/script_args.oversample):]
        # TODO will need to sanity check this since its weirder
        for k in keepvars: 
            keep_inds.extend(list(range(k*script_args.oversample, (k+1)*script_args.oversample)))
    elif 'var_min' in script_args.rollout_strategy:
        keep_inds = []
        keepvars = list(np.argsort(varlist))[:int(keeplen/script_args.oversample)]
        # TODO will need to sanity check this since its weirder
        for k in keepvars: 
            keep_inds.extend(list(range(k*script_args.oversample, (k+1)*script_args.oversample)))

    # batch format (rollouts separated by minibatches, each minibatch has first half preferred, 2nd half disp)
    if dpoplus:
        # This is a hack
        if script_args.oversample==1:
            keep_inds = keep_inds[:int(len(keep_inds)/2)]
        # set things up to handle minibatching correctly 
        new_inds = []
        midpoint = int(len(keep_inds)/2)
        permid = int(script_args.mini_batch_size/2)
        for i in range(0, midpoint, permid):
            # TODO may need to handle "repeats" where argmin, argmax get the saem thing
            new_inds.extend(keep_inds[i:i+permid]+keep_inds[midpoint+i:midpoint+i+permid])
        keep_inds = new_inds
    return keep_inds

""" Method to get rollouts, specifically, it supports the ability to have multiple sources of rollouts"""
def get_rollouts(ppo_trainer, question_tensors, output_length_sampler, script_args, generation_kwargs, tmpmodel, ratio):
    responses = []
    inrange = list(range(len(question_tensors)))
    kl_mask = [1]*len(question_tensors)
    # do stuff based on generation from ratio thing, TODO could support more interesting stuff if this works?
    if ratio>0:
        if tmpmodel!='orig':
            get_unwrapped(ppo_trainer).set_adapter(tmpmodel)
        random.shuffle(inrange)
        if tmpmodel=='orig':
            with ppo_trainer.optional_peft_ctx():
                responses.extend(ppo_trainer._generate_batched(
                    ppo_trainer.model,
                    [question_tensors[q] for q in inrange[:int(ratio*len(question_tensors))]],
                    length_sampler=output_length_sampler,
                    batch_size=script_args.gen_bsize,
                    return_prompt=False,
                    **generation_kwargs,
                ))
        else:
            responses.extend(ppo_trainer._generate_batched(
                ppo_trainer.model,
                [question_tensors[q] for q in inrange[:int(ratio*len(question_tensors))]],
                length_sampler=output_length_sampler,
                batch_size=script_args.gen_bsize,
                return_prompt=False,
                **generation_kwargs,
            ))
        for j in inrange[:int(ratio*len(question_tensors))]:
            kl_mask[j]=0
        print("this one has been generated successfully")
    
    if ratio<1:
        get_unwrapped(ppo_trainer).set_adapter("default")
        # use current rollout for the reset
        responses.extend(ppo_trainer._generate_batched(
            ppo_trainer.model,
            [question_tensors[q] for q in inrange[int(ratio*len(question_tensors)):]],
            length_sampler=output_length_sampler,
            batch_size=script_args.gen_bsize,
            return_prompt=False,
            **generation_kwargs,
        ))
    # put back in the correct order
    results = [None]*len(question_tensors)
    for i in range(len(responses)):
        results[inrange[i]] = responses[i]
    # kl_mask will tell us which outputs are weird, will need KL term turned off to keep things from going crazy
    return results, kl_mask
    
def get_unwrapped(ppo_trainer):
    return ppo_trainer.accelerator.unwrap_model(ppo_trainer.model).pretrained_model


# TODO try with and without context (will give bonus to newer contexts)
def update_tokdict_bonuses(responses, tokdict): 
    # add based on unique things where a word occurs
    resps = [set(word_tokenize(resp)) for resp in responses]
    bonuses = []
    base = len(responses)
    if len(tokdict.values())>0:
        base = base + max(tokdict.values())
    for r in resps: 
        bonus = 0
        for word in r: 
            if word not in tokdict.keys(): 
                tokdict[word] = 0
            # NOTE order sort of matters here, don't give bonus at the very beginning, not sure if matters
            tokdict[word] += 1
            bonus+= (1-math.sqrt(tokdict[word]/base))
        bonuses.append(bonus)
    # TODO play around with scaling a bit
    return bonuses

# Function to process reward scores
def process_reward(texts, rmname, reward_model, script_args, response_tensors, metadata):
    sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 8, "truncation": True}
    # NOTE RM score is based on API endpoint (rmname)
    if "http" in rmname: 
        # TODO metadata to api calls? 
        rewards = [s for s in get_scores_from_api(texts, rmname)]
    # using some kind of simple function-based reward
    elif "function" in rmname: 
        rewards = [s for s in get_synth_rewards(texts, rmname, metadata)]
    else: # otherwise based on in-process RM
        pipe_outputs = reward_model(texts, **sent_kwargs)
        if script_args.len_only>0:
            # the reward is just a weird function thing, you set the max
            rewards = [lensco(len(response)/script_args.len_only) for response in response_tensors]
        else:
            # TODO length constraints, other fancy stuff gets added in here
            rewards = [output[0]["score"]-script_args.reward_baseline for output in pipe_outputs]
    if script_args.keep_long>0:
        radds = [(-20 if len(word_tokenize(t.strip()))<script_args.keep_long else 0) for t in texts]
        rewards = [rewards[i]+radds[i] for i in range(len(radds))]
    return rewards

def train_loop(script_args, ppo_trainer, reward_model, tokenizer, qaform):
    
    if script_args.replay_freq<=0:
        script_args.replay_freq=1000000
    
    # global likemod, liketok, slikemod, sliketok
    # store this for use later
    replay_buffer = {'responses': [], 'rewards':[], 'questions':[], 'klmask':[]}
    
    # the `generate` function of the trained model.
    generation_kwargs = { 
        "min_length": 20, "top_k": 0.0,"top_p": 1, "do_sample": True, "temperature":script_args.temperature #"pad_token_id": tokenizer.pad_token_id, # "eos_token_id": 100_000,
    }
    curstrat = -1
    # every time we hit a new index in stratchange list, rollout strategy changes (new checkpoint on top of initial)
    sjson = {'intervals':[100000]}
    if (script_args.generators_json !=None) and (script_args.generators_json != "none"):
        with open(script_args.generators_json) as f:
            sjson = json.load(f)
            
    # TODO add option to shuffle before oversample generation
    current_device = Accelerator().local_process_index
    
    print(reward_model, " is RM")
    if "contrastivedistill" in script_args.reward_model_name:
        print("loading the RM")
        rw.likemod = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b", device_map=current_device, torch_dtype=torch.bfloat16).eval()
        rw.liketok = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
        rw.slikemod = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", device_map=current_device, torch_dtype=torch.bfloat16).eval()
        rw.sliketok = AutoTokenizer.from_pretrained("facebook/opt-125m")
    if "eurusrm" in script_args.reward_model_name:
        print("eurus RM loading")
        # RM with its own custom code setup
        rw.slikemod = AutoModel.from_pretrained("openbmb/Eurus-RM-7b", device_map=current_device, torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
        rw.sliketok = AutoTokenizer.from_pretrained("openbmb/Eurus-RM-7b")

    rollout_tokens_dict = {}
    min_len = script_args.max_length-2
    if script_args.trl_weird==1:
        # TODO add in temperature here
        generation_kwargs = {
            "top_k": 0.0,"top_p": 1, "do_sample": True, "pad_token_id": tokenizer.pad_token_id, "eos_token_id": 100_000,
        }
        min_len = 32

    # HACK since I don't like how they set up RL code length stuff
    output_length_sampler = LengthSampler(min_len, script_args.max_length)

    # compute moving averages over last 10 steps
    running_means = []
    running_stds = []
    rmname = script_args.reward_model_name
    if 'einstein' in rmname:
        generation_kwargs['logits_processor']=[EinsteinLogitsProc(tokenizer, 4, 2)]
    tot_rollouts = 0
    
    dpoplus = script_args.kl_penalty=="dpoplus"
    # TODO set up some arg assertions here for rollout strategies, put earlier in training
    if ("normal" not in script_args.rollout_strategy):
        assert script_args.oversample>1
    
    tmpmodel = None
    ratio = 0
    normalsteps = 0
    cur_replays = 0
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        if sjson['intervals'][curstrat+1]<epoch:
            print("HOORAY! we're adding another rollout sampler into the mix")
            curstrat+=1
            # make an exception for just sft case
            if sjson['checkpoints'][curstrat]!='orig':
                get_unwrapped(ppo_trainer).load_adapter(sjson['checkpoints'][curstrat], sjson['checkpoints'][curstrat], is_trainable=False)
                get_unwrapped(ppo_trainer).to(current_device)
            ratio = sjson['ratios'][curstrat]
            tmpmodel = sjson['checkpoints'][curstrat]
        if epoch >= script_args.steps:
            break
        
        if (normalsteps+1)%script_args.replay_freq==0:
            print("starting replay")
            cur_replays = script_args.replay_updates
            normalsteps+=1

        # this will now allow us to do multiple replay steps for every normal step
        doreplay = cur_replays>0
        cur_replays = cur_replays - 1
        if doreplay==False:
            print("normal step")
            normalsteps+=1

        print(batch.keys())
        # get rid of shared context within individual things? 
        if script_args.rollout_strategy=="random":
            inds = list(range(len(batch['input_ids'])))
            random.shuffle(inds)
            batch['input_ids'] = [batch['input_ids'][i] for i in inds]
            batch['query'] = [batch['query'][i] for i in inds]

        question_tensors = batch["input_ids"]
        
        new_questions = []
        new_qs = {k:[] for k in batch.keys()}
            
        # we're doing something to oversample 
        if script_args.oversample>1: 
            # this should do the trick, since dpo batch sizes are always in higher terms?
            qlim = int(len(question_tensors)/2) if (dpoplus) else len(question_tensors)
            for i in range(qlim): 
                new_questions.extend([question_tensors[i]]*script_args.oversample)
                for k in new_qs.keys():
                    new_qs[k].extend([batch[k][i]]*script_args.oversample)
            assert len(new_questions)==qlim*script_args.oversample
        
            question_tensors = new_questions
            batch = new_qs
            
            if 'normal' in script_args.rollout_strategy: 
                question_tensors = question_tensors[:script_args.batch_size]
                for k in batch.keys():
                    batch[k] = batch[k][:script_args.batch_size]
        
        if epoch == 0:
            # SANITY CHECKING
            print("PPO input")
            print(tokenizer.batch_decode(question_tensors, skip_special_tokens=True))
    
        with torch.no_grad():
            if doreplay:
                # NOTE we don't need to waste time on another rollout, just replace with old data
                # TODO sanity check that these are paired. If not move things earlier into the pipeline
                inds = list(range(0, len(replay_buffer['responses']), 2))
                random.shuffle(inds)
                inds = inds[:int(script_args.batch_size/2)] # should be fine for PPO too?
                newinds = []
                for tmpind in inds:
                    newinds.extend([tmpind, tmpind+1])
                response_tensors = [replay_buffer['responses'][tmpind] for tmpind in newinds]
                question_tensors = [replay_buffer['questions'][tmpind] for tmpind in newinds]
                kl_mask = [replay_buffer['klmask'][tmpind] for tmpind in newinds]
                rewards = [replay_buffer['rewards'][tmpind] for tmpind in newinds]
            else:
                # TODO more sophisticated sampling logic (maybe get some API call action, using dataset, etc.)
                response_tensors, kl_mask = get_rollouts(ppo_trainer, question_tensors, output_length_sampler, script_args, generation_kwargs,tmpmodel,abs(ratio))
                tot_rollouts+=len(response_tensors)
            # model logprobs can be used directly as rewards
            
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        if (script_args.self_reward_steps>0) or script_args.log_genaccs: 
            # TODO can get rid of this in the doreplay case maybe? do that if it's taking up too much time or smth
            # TODO need to throw mini-batch-size thing on this
            with torch.no_grad():
                oldinps = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
                # TODO can't handle mini-batching yet
                tmppinps = tokenizer([batch['response'][i]+oldinps[i] for i in range(len(oldinps))], padding=True, truncation=True, return_tensors="pt").to(current_device)
                # DO the "batched forward pass from their thing, only risk is it may not work"
                newlogits = ppo_trainer.accelerator.unwrap_model(ppo_trainer.model)(input_ids=tmppinps.input_ids, attention_mask=tmppinps.attention_mask)[0]
                with ppo_trainer.optional_peft_ctx():
                    if ppo_trainer.ref_model is not None:
                        reflogits = ppo_trainer.accelerator.unwrap_model(ppo_trainer.ref_model)(input_ids=tmppinps.input_ids, attention_mask=tmppinps.attention_mask)[0]
                    else:
                        # we're using the basic peft model in this case? 
                        reflogits = ppo_trainer.accelerator.unwrap_model(ppo_trainer.model)(input_ids=tmppinps.input_ids, attention_mask=tmppinps.attention_mask)[0]
                # can just use the things directly since it's only pairwise
                selfrewards = get_batch_logps(newlogits, tmppinps.input_ids, label_pad_token_id=tokenizer.pad_token_id)
                # normalize by reference (otherwise length will dominate)
                selfrewards = selfrewards - get_batch_logps(reflogits, tmppinps.input_ids, label_pad_token_id=tokenizer.pad_token_id)
                selfrewards = selfrewards.tolist()

                print("got selfrewards: ", selfrewards)
            
        if "einstein" in rmname:
            batch['response'] = ["\n".join(r.split("\n")[:2]) for r in batch['response']]
            response_tensors = [tokenizer(r, return_tensors="pt").input_ids.squeeze(0) for r in batch['response']]
        if "math" in script_args.output_dir: 
            print("math proc")
            # manually make sure we don't have a ton of extra stuff for each one of these
            # NOTE got rid of math hack for now
            # batch['response'] = ["=".join(batch['response'][indval].split("=")[:batch['query'][i].count("(")+1]) for indval in range(len(batch['response']))]
        
        # let's manually avoid using stuff that needs to get omitted
        if script_args.omit_long==1:
            omit_long(tokenizer, response_tensors, question_tensors, batch)
    
        if epoch == 0:
            # SANITY CHECKING
            print("QAForm Input Example:")
            print(qaform(batch['query'][0], batch['response'][0]))

        # Get RM score, NOTE that the input formatting is reward model specific
        texts = [qaform(q, r) for q, r in zip(batch["query"], batch["response"])]
        if dpoplus:
            if batch['query'][0]!=batch['query'][1]:
                print("WARNING, PAIRS ARE NOT MATCHING, SOMETHING IS WRONG")
        # we need to use 2 reward sources simultaneously
        # NOTE this code may be buggy
        if "rewardratios" in sjson.keys():
            print("using reward ratio split")
            inds = list(range(len(texts)))
            # shuffle up so that scoring is split up
            if dpoplus==False:
                inds.shuffle()
            ttmps = [texts[i] for i in inds]
            lval = 0
            currats = sjson['rewardratios'][curstrat]
            rtmps = []
            # only need 2 for now, but support more just in case
            for r in range(len(currats)):
                lim = lval+int(currats[r]*len(ttmps))
                cstrat = sjson['rsources'][r]
                if currats[r]==0:
                    continue
                # HACK default always needs to come first for scaling to work
                if cstrat=="default":
                    rtmps.extend(process_reward(ttmps[lval:lim], rmname, reward_model, script_args, response_tensors, batch))
                else:
                    # TODO this only works for incorporating gold / external apis atm (which Ig covers everything)
                    nrewards = process_reward(ttmps[lval:lim], cstrat, reward_model, script_args, response_tensors, batch)
                    # need to get the scale to match
                    nrewards = [n-mean(nrewards) for n in nrewards]
                    # HACK TODO make the value of 2 a hyperparam
                    nrewards = [n*2+mean(rtmps) for n in nrewards]
                    rtmps.extend(nrewards)
                lval = lim
                # undo the shuffling so that stuff matches
            rewards = [rtmps[inds.index(i)] for i in range(len(inds))]
        else:
            if doreplay==False: # don't get reward on replay
                rewards = process_reward(texts, rmname, reward_model, script_args, response_tensors, batch)
        rewards = [float(r) for r in rewards]
        print("mean gold reward is", mean(rewards))
        
        if (script_args.self_reward_steps>0) and doreplay==False: 
            racc = []
            # check self-reward acc
            for itmp in range(0, len(selfrewards), 2):
                if rewards[itmp]!=rewards[itmp+1]:
                    racc.append(1 if (rewards[itmp]>rewards[itmp+1])==(selfrewards[itmp]>selfrewards[itmp+1]) else 0)
            if len(racc)>0:
                print('accuracy of self-reward is ', mean(racc))
            if (selfrewards[0]==0):
                print("warning, self reward is 0, indicating that ref mode and base model are the same heer")
            assert  (len(selfrewards)==len(rewards))
            # NOTE code for prepping self-reward stuff for updates
            if script_args.self_reward_rollouts<=tot_rollouts:
                # last N - k things get their labels readjusted (keep the gold for the rest)
                for k in range(script_args.self_reward_steps*2, len(rewards)):
                    rewards[k] = selfrewards[k]
                # HACK handle the trainer kwargs correctly, this is janky
                ppo_trainer.config.kl_penalty="selfreward"
                ppo_trainer.config.ratio_threshold=script_args.self_reward_steps*2

        # TODO make this argument, find a way to increase how much it happens
        if (script_args.replay_freq>0) and (doreplay==False):
            print("adding replay batches")
            # NOTE handle self-reward, gold case
            lim = script_args.self_reward_steps*2 if script_args.self_reward_steps>0 else len(response_tensors)
            replay_buffer['responses'].extend(response_tensors[:lim])
            replay_buffer['questions'].extend(question_tensors[:lim])
            assert replay_buffer['questions'][0].tolist()==replay_buffer['questions'][1].tolist()
            replay_buffer['klmask'].extend(kl_mask[:lim])
            replay_buffer['rewards'].extend(rewards[:lim])
            for r in replay_buffer.keys():
                # keep a certain level of recency
                replay_buffer[r] = replay_buffer[r][-script_args.batch_size*script_args.replay_batches:]
                
        # different strategies on how to deal with oversampling, make sure to prop through all the variables to avoid errors
        keep_inds = keep_strat(script_args, rewards, list(range(len(rewards)))) # default
        if script_args.save_rollouts:
            roll_dict = {'inputs':batch['query'], 'outputs':batch['response'], 'rewards':rewards, 'keepinds':keep_inds, 'selfscos':selfrewards if script_args.log_genaccs else []}
        if epoch==0:
            print(rewards)
            print(keep_inds)
        print("KEEP INDS: ", len(keep_inds))
        print("RM INIT", mean(rewards))
        print('RM STDEV INIT', stdev(rewards))
        rewards = [rewards[k] for k in keep_inds]
        # should handle any extra stuff
        for ky in batch.keys():
            batch[ky] = [batch[ky][k] for k in keep_inds]
        response_tensors = [response_tensors[k] for k in keep_inds]
        question_tensors = [question_tensors[k] for k in keep_inds]
        kl_mask = torch.tensor([kl_mask[k] for k in keep_inds]).to(current_device)
        
        print("RM MEAN", mean(rewards))
        print('RM STDEV', stdev(rewards))
        rewards = [torch.tensor(r).to(current_device) for r in rewards]
        
        # using running mean and running stdev to do stuff if needed
        rws = [float(f) for f in rewards]
        running_means.append(mean(rws))
        running_stds.append(stdev(rws))
        meanval = mean(running_means[-10:])
        sigma = mean(running_stds[-10:])
        
        logrewards = []
        logmean = []
        # we can set negative ratio for 'contrastive' reward that cancels out reward from certain outputs
        for i in range(len(kl_mask)):
            # only log using rewards from the properly sampled outputs, will want values to be subbed in with mean
            if kl_mask[i]==1:
                logrewards.append(float(rewards[i].item()))
                logmean.append(float(rewards[i].item()))
            else: 
                logrewards.append(-1000)
                if ratio<0:
                    # TODO test making 0 vs making negative
                    rewards[i] = -1*sigma*rewards[i]

        if len(logmean)>0:
            lmean = mean(logmean)
        else:
            lmean = float(0)
        # don't use weird values, or counter-examples
        for i in range(len(logrewards)):
            if logrewards[i]==-1000:
                logrewards[i] = lmean
        
        if script_args.len_penalty==1:
            # basically if you go 50 tokens out (hit limit), then you get penalized by 1 stdev of RM score
            penalties = [2*sigma*(1 - 0.01*len(r)) for r in response_tensors]
            rewards = [rewards[i]+penalties[i] for i in range(len(rewards))]
            if epoch<10:
                print(rewards, " : penalties : ", penalties)

        # reward scaling from secrets of PPO
        if script_args.scale_reward==1:
            rewards = [torch.clip((rw - meanval)/sigma, -0.3, 0.3) for rw in rewards]
            if epoch<10:
                print(rewards)
                
        if script_args.tok_bonus_ratio>0:
            bonuses = update_tokdict_bonuses(batch['response'], rollout_tokens_dict)
            bonuses = [b*sigma*script_args.tok_bonus_ratio for b in bonuses]
            print("Bonuses: ", bonuses)
            # TODO need to tune the scaling stuff a little bit
            rewards = [rewards[i]+bonuses[i] for i in range(len(rewards))]
            
        stats = ppo_trainer.step(question_tensors, response_tensors, rewards, kl_mask=kl_mask)
        
        if script_args.save_rollouts: 
            # NOTE removing stats so that log_dicts aren't taking a ton of space
            roll_dict['step'] = epoch
            rollfile = script_args.output_dir.replace("checkpoints/", "results/rollouts/")
            while rollfile[-1]=='/':
                rollfile = rollfile[:-1]
            rollfile = rollfile+".jsonl"
            append_dict_to_jsonl(roll_dict, rollfile)
            
        ppo_trainer.log_stats(stats, batch, logrewards)
        if script_args.save_freq and epoch and epoch % script_args.save_freq == 0:
            create_missing_folders_for_file(script_args.output_dir + f"step_{epoch}"+"/")
            ppo_trainer.save_pretrained(script_args.output_dir + f"step_{epoch}")
            
        if script_args.self_reward_steps>0: 
            racc = []
            for itmp in range(0, len(selfrewards), 2):
                if rewards[itmp]!=rewards[itmp+1]:
                    racc.append(1 if (rewards[itmp]>rewards[itmp+1])==(selfrewards[itmp]>selfrewards[itmp+1]) else 0)
            print('accuracy of self-reward is ', mean(racc))
            # assert selfrewards[0]<=0 
            if script_args.self_reward_rollouts<=tot_rollouts:
                # first k things get their labels readjusted
                for k in range(script_args.self_reward_steps*2):
                    rewards[k] = selfrewards[k]
                # HACK handle the trainer kwargs correctly, this is janky
                ppo_trainer.config.kl_penalty="dpoplus"
                ppo_trainer.config.ratio_threshold=0
                tot_rollouts=0
                
        