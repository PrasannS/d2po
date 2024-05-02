# D2PO: Discriminator-Guided DPO with Response Evaluation Models

This repo contains code and instructions for reproducing experiments in the paper "D2PO: Discriminator-Guided DPO with Response Evaluation Models", by Prasann Singhal, Nathan Lambert, Scott Niekum, Tanya Goyal, and Greg Durrett. We propose a new data-efficient approach for optimizing online with preference sources.  

## Installation

First make sure to set up an environment with Python 3.10, you can then get the necessary installations with 

```
# install normal requirements
pip install -r requirements.txt
# editable install of rlhf_utils with necessary helper code
cd rlhf_utils
pip install -e .
cd ..
```

## Setting up Data
We construct datasets for our different preference tasks for RM and policy training. We plan to release files with our used datasets in the near future, however if you want to apply our code to a custom task: 
- for preferences : you can simply create a huggingface dataset with row format of "question", "response_j" (for preferred) and "response_k" (for dispreferred output)
    - not used in the work but you can add a "magnitude" row for better RMs if you have access to those (will need to use the right flag in training) 
- for rollouts : you can simply create a huggingface dataset with rows that contain the row "question"
- for creating things from scratch:
    - you can take this starting dataset: https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized (or some other one of your choosing), and rescore pairs with our different gold functions using the get_synth_rewards function in src/eval/rewards.py

## SFT Models
We use [TULU-2-7B](https://huggingface.co/allenai/tulu-2-7b)  for our realisitic experiments, a fine-tuned [OPT-1.3b](https://huggingface.co/facebook/opt-1.3b) on our math experiments, and normal [OPT-125m](https://huggingface.co/facebook/opt-125m) for all other experiments. 

## Training a Reward Model 

We include the generic script for training a reward model:

You can run it as follows (make sure you have your preference data / held out set in outputs/data/{goldfunction}/): 
```
# make sure to set to number of GPUs of your choice
export CUDA_VISIBLE_DEVICES=0,1
# some default hyperparams
export LR=1e-4
export NOLORA=False
export REINIT=False
export LTYPE="normal"
export EVFIRST=0
export BSIZE=2
# run script
export BASEMODEL={path to SFT model}

sh script/train_rm.sh {gold function} {dataset file name} {eval set name} {process id, can just be 12450} {run name tag}
```

Some gold functions are: 
- bagofwords: word collector
- unique_nns: unique nouns
- contrastivedistill: contrastive distill
- (careful, this costs money) ultrafeedbackgold: gpt-4 ultrafeedback
- eurusrm: eurus rm
- math: math
- more in rewards.py, can define additional ones straightforwardly as well

Once you have a checkpoint that you're happy with, you can merge it (put it in outputs/models/{gold reward name})
```
python src/adapter.py \
    --adapter_model_name="{PATH_TO_SAVE_CHECKPOINTS}/checkpoint_{BEST_CHECKPOINT}" \
    --base_model_name="{PATH_TO_SFT_MODEL}" \
    --output_name="{REWARD_MODEL_NEW_PATH}"
```

## DPO

We include the generic script for training with DPO:

You can run it as follows (make sure you have your preference data / held out set in outputs/data/{goldfunction}/): 
```
export CUDA_VISIBLE_DEVICES=0
export BASEMODEL={starting point reference / SFT model}
export BETA=0.05
sh script/train_dpo.sh {gold reward id} {train dataset (same format as RM)} {eval dataset (same format as RM)} {run name tag} {process id}

```
    
## Training with RLHF 

**HACK**
Once you're done installing things: 

1. You'll want to navigate to the source code of your TRL install, and override trainer/ppo_trainer.py with the code in utils/misc/dpoplus_trainer.py
2. Do the same with ppo_config.py, replacing it with utils/misc/config_ppo.py

This will setup OPO code and some other changes to TRL source code in order to get our approach up and running. We plan to fix this with monkey-patching in the near future. 

If you want to do normal training (OPO with a gold function or a static reward model), it's pretty easy from there: 

```
export CFG=src/configs/ppo_2gpu.yaml
export SUPDATES=10000000
export SEED={seed}
export KEEPLONG={min length, usually doesn't affect anything}
export MLEN={max length, set to 50 for most synth settings, 256 on realistic}
export BASEMODEL={starting SFT model (can use a DPO-based model if you want)}
# how many epochs per policy batch (try increasing to tune hyperparams on new settings)
export PPOUPDATES=1
# batch size per process
export DPOBATCHSIZE=32
# mini batch size (can reduce to fit things on GPU)
export MBSIZE=32
# generation batch size (can reduce to fit things on GPU)
export GBSIZE=32
# train steps (in terms of modified PPOTrainer)
export STEPS=2000

# set GPUs 
export CUDA_VISIBLE_DEVICES=0,1
sh script/dpoplus_script.sh {gold reward function} {path to prompt data, can also use "ultra"} {RM name} {unique process ID} {run name tag}
```

RM name should be set to either your reward model in outputs/models/{gold function name} for a static RM, you can also set it to "function{gold function name}" if you want to do OPO with gold. 

You can check out src/utils/args/ppo_args.py for lots of other hyperparams / implemented configurations if you're curious. 

## D2PO

If you want to run D2PO, the script looks a bit more complicated: 

```
export CFG=src/configs/ppo_2gpu.yaml
export SUPDATES=10000000
export SEED={seed}
export KEEPLONG={min length, usually doesn't affect anything}
export MLEN={max length, set to 50 for most synth settings, 256 on realistic}
export BASEMODEL={starting SFT model (can use a DPO-based model if you want)}
# how many epochs per policy batch (try increasing to tune hyperparams on new settings)
export PPOUPDATES=1
# batch size per process
export DPOBATCHSIZE=32
# mini batch size (can reduce to fit things on GPU)
export MBSIZE=32
# generation batch size (can reduce to fit things on GPU)
export GBSIZE=32
# train steps (in terms of modified PPOTrainer)
export STEPS=2000

# D2PO specific hparams
export ATYPE={conf/rand} # for confidence vs random sampling
export APBSIZE=16 {how many examples for RM to take in per batch}
export GREWARD={gold reward name}
export ULR=1e-4 {RM learning rate}
export UEPOCHS=4 {how many times RM updates on each amount of data}
export SFREQ=25 {checkpoint frequency}
# main hyperparams
export SAMPN=$((256)) # how many policy preferences (*2) before collecting relabels
export RELABELS=$((16)) # how many gold preferences to get whenever we hit SAMPN rollouts

# start API with D2PO RM updating live
nohup sh script/newupdateapi.sh "" {starting rm name (in outputs/models/{gfunct})} {job name tag} 5007 &

# set GPUs 
export CUDA_VISIBLE_DEVICES=0,1
sh script/dpoplus_script.sh {gold reward function} {path to prompt data, can also use "ultra"} "http://127.0.0.1:5007/train" {unique process ID} {run name tag}

# kill api at the end otherwise it runs forever
jobs
pkill -f {run name tag}
jobs
```

See different hyperparameters and options in src/utils/args/api.py to try different configurations out. 

## Evaluation

Once you have your desired PPO checkpoint, you can do inference to get evaluation results (gold reward): 

```
# Define a function to run the script with different inputs
run_script() {
    # NOTE that we need to feed things in a specific format
    CKPT_FILE="outputs/checkpoints/${1}/${2}${3}${4}"
    OUTPUT_DIR="outputs/results/genouts/${1}/${2}${4}"
    
    python -u src/eval/generate_outs.py \
        --basemodel="$BASEMODEL" \
        --dset="$DSET" \
        --ckname="$CKPT_FILE" \
        --fname="$OUTPUT_DIR" \
        --bottom=$BOTTOM --top=$TOP  \
        --bsize=$BSIZE \
        --maxlen=$MLEN \
        --genbatch=$GBATCH

    python -u src/evalgold.py  --fname="${OUTPUT_DIR}.jsonl" --gfunct="${1}"
}

BOTTOM=0
TOP=200 # {which examples from held out set to use}
MLEN=50 # {max length}
BSIZE=1 # {how many outputs per input}
# held out dataset
DSET={path to held out eval set}

BASEMODEL={path to base model used as starting point}
GBATCH=32 # {batch size}

export CUDA_VISIBLE_DEVICES=1
# generate over many checkpoints
for i in $(seq 100 100 2000)
do
  run_script {gold reward function} "ppo_{run name tag}" "/step_" "$i"
done

```

This will generate output files with gold reward scores that you can then calculate evaluation metrics over (e.g. mean gold reward). 

## Coming Soon!

- We plan on releasing trained models, as well as cleaning up code and scripts further to make things easier to run.

## Citation 

{coming soon}
