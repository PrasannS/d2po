# GENERAL CODE FOR RLHF TRAINING ON OUR DIFFERENT SETTINGS

import os
from tqdm import tqdm
from transformers import HfArgumentParser, AutoModelForCausalLM
from trl import  PPOTrainer, set_seed
from datasets import concatenate_datasets

from utils.rl_utils import (
    load_models,
    train_loop
)

from utils.args.ppo_args import PPOArguments
from rlhfutils.data import (
    build_wgpt_promptdata,
    build_rlcd_promptdata,
    build_stack_promptdata,
    build_apf_promptdata,
    build_ultra_promptdata,
    build_custom_promptdata, 
    collator,
    qaform,
    anscat, 
    webgpt_template, 
    tulu_pf
)

os.environ["WANDB_TAGS"] = "[\"llamatrl\"]"
tqdm.pandas()

parser = HfArgumentParser(PPOArguments)
script_args: PPOArguments = parser.parse_args_into_dataclasses()[0]

set_seed(script_args.seed)

if script_args.output_dir[-1]!="/":
    script_args.output_dir = script_args.output_dir+"/"

print("over here")
# NOTE special case if using an api endpoint
if "http" in script_args.reward_model_name:
    config, tokenizer, model, optimizer = load_models(script_args, "ppo")
    reward_model = None
elif "function" in script_args.reward_model_name:
    config, tokenizer, model, optimizer = load_models(script_args, "ppo")
    reward_model = "function"
else:
    # NOTE handle loading everything in, since hyperparams are same for every setting more or less
    config, tokenizer, model, optimizer, reward_model = load_models(script_args)

print("loaded models")
rmformat = qaform
if "wgpt" == script_args.dataset_name:
    dataset = build_wgpt_promptdata(tokenizer)
    # TODO the ones below this
elif "rlcd" in script_args.dataset_name:
    dataset = build_rlcd_promptdata(tokenizer, script_args.dataset_name)
    rmformat = anscat  # NOTE RLCD RM has a different prompt template depending on the model, this is a bit ad-hoc
elif "stack" == script_args.dataset_name:
    dataset = build_stack_promptdata(tokenizer)
    rmformat = anscat
elif "apfarm" == script_args.dataset_name:
    dataset = build_apf_promptdata(tokenizer)
    rmformat = anscat
# TODO fix ultrachat datset issue
elif ("ultra" == script_args.dataset_name) or ("ultra/" in script_args.dataset_name) :
    print("NOTE we're not using custom data, we're using default ultafeedback here")
    # TODO maybe unify original prompt format? 
    template = tulu_pf if "tulu" in script_args.model_name else webgpt_template
    print("MODEL NAME IS (CHECK IF WE SHOULD BE USING TULU) ", script_args.model_name)
    dataset = build_ultra_promptdata(tokenizer, script_args.dataset_name, template)
else: 
    pftmp = "default"
    mdatatmp = []
    if "einstein" in script_args.dataset_name: 
        print("einstein data format")
        pftmp = 'einstein'
        mdatatmp = ['sol_rows', 'response_j']
    elif "distil" in script_args.dataset_name or "math" in script_args.dataset_name: 
        pftmp = 'onlyans'
    if "tulu" in script_args.model_name: 
        print('using TULU template')
        pftmp = 'tulu'
        # mdatatmp = ['response_k', 'response_j']
    # keep track of solution rows
    dataset = build_custom_promptdata(tokenizer, script_args.dataset_name, pftmp, mdatatmp)
if ("math" in script_args.reward_model_name) and ("function" in script_args.reward_model_name): 
    print("beware, using math format")
    rmformat = anscat
# if ("distil" in script_args.reward_model_name) and ("function" in script_args.reward_model_name): 
#     print("beware, using concat format")
#     rmformat = anscat
print(dataset[0])
# if len(dataset)<(script_args.steps*2*script_args.batch_size):
#     dataset = concatenate_datasets([dataset]*int(1 + (script_args.steps*2*script_args.batch_size)/len(dataset)))
#     print('extended dataset to size: ', len(dataset))

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
if script_args.self_reward_steps>0:
    print("loading refmod")
    refmod = model[1]
    model = model[0]
    
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model= refmod if  script_args.self_reward_steps>0 else None,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer
)

# TODO customize for different RM code, and different RM input formats
# Run RL pipeline now
train_loop(script_args, ppo_trainer, reward_model, tokenizer, rmformat)