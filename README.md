# D2PO: Discriminator-Guided DPO with Response Evaluation Models

This repo contains code and instructions for reproducing experiments in the paper "D2PO: Discriminator-Guided DPO with Response Evaluation Models", by Prasann Singhal, Nathan Lambert, Scott Niekum, Tanya Goyal, and Greg Durrett. We propose a new data-efficient approach for optimizing online with preference sources.  

NOTE: this readme is a work in progress, this message should be gone in a couple of days with an updated readme soon. 

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
    
## Training with RLHF 

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

## Evaluation

Once you have your desired PPO checkpoint and reward model, you can do inference to get evaluation results. If you want to use the OpenAI API-based AlpacaFarm evals, you'll need to put an OpenAI API key in secret/openaikey.txt, otherwise, that isn't necessary. 

First, you need to generate a set of outputs from the PPO checkpoint: 

```
export CUDA_VISIBLE_DEVICES=0
python -u generate_outs.py \
    "{SFT_MODEL}" \
    {"webgpt", "rlcd", "stack"} \
    "{PATH_TO_SAVE_CHECKPOINT}/step_{PPO_CHECKPOINT_STEP}" \
    "{OUTPUT_NAME}" \
    0 500  \
    {SAMPLES_PER_PROMPT}
```

0 and 500 are the bottom and top of the eval dataset subset that you want to do eval with (note that seeds are fixed for reproducibility). Samples_per_prompt can be set to 1 in most cases. This will generate a file generated_{OUTPUT_NAME}.jsonl containing inputs and outputs for a fixed prompt set from the desired generation model with default decoding hyperparameters. You can set the path parameter to just "orig" if you want to generate from just the SFT model. 

Once you have your generated output files, you can follow eval/simulated_prefs.ipynb for simulated preference eval. If you want to score the outputs with your original reward model, you can do so by running: 

```
python -u rmsco_outs.py \
    --rmname="{PATH_TO_RM}" \
    --inpf="{GENERATION_FILE}" \
    --device {GPU_ID} \
    --lim {TOP_INDEX_TO_SCORE} \
    --shuffle 0
```

Which you can then use to reproduce correlation numbers and reward numbers (see eval/measure_intrinsic.ipynb). 

## Coming Soon!

- We plan on releasing trained models, as well as more scripts to make things easier to run. 
- Additional Follow-Up Experiments
- Blog Post with more analysis on the paper
- 
## Citation 

{coming soon}
