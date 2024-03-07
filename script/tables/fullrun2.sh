# SH File for running the whole shebang, multiple jobs

export CFG=src/configs/ppo_2gpu.yaml
export STEPS=2000
export LABTHRESH=0.3
export SUPDATES=10000000
export SEED=0
export KEEPLONG=0
export MLEN=50

defaults() {
    export BASEMODEL="outputs/models/bagofwords/bowtiny_dpo"
    export REDOBATCH=1
    export LABRATIO=0.25
    export ATYPE="rand"
    export ULR=1e-4
}

# BEST TECHNIQUE (ALL IN ONE)
defaults
export ATYPE="conf"
export REDOBATCH=15

export CUDA_VISIBLE_DEVICES=5
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/updateapi.sh "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "conf_all" 5002 & 
# Other commands
export CUDA_VISIBLE_DEVICES=6,7
sh script/dpoplus_script.sh "bagofwords" "ultra" "http://127.0.0.1:5002/train" 29517 "conf_all"
jobs
pkill -f "conf_all"
jobs

defaults
# RAND SUB_SAMP + REPLAY (5 batch)

export REDOBATCH=5

export CUDA_VISIBLE_DEVICES=5
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/updateapi.sh "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "rand_subsamp_replay5" 5002 & 
# Other commands
export CUDA_VISIBLE_DEVICES=6,7
sh script/dpoplus_script.sh "bagofwords" "ultra" "http://127.0.0.1:5002/train" 29517 "rand_subsamp_replay5"
jobs
pkill -f "rand_subsamp_replay5"
jobs


# RAND SUB_SAMP + REPLAY (15 batch)

defaults
export REDOBATCH=15

export CUDA_VISIBLE_DEVICES=5
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/updateapi.sh "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "rand_subsamp_replay15" 5002 & 
# Other commands
export CUDA_VISIBLE_DEVICES=6,7
sh script/dpoplus_script.sh "bagofwords" "ultra" "http://127.0.0.1:5002/train" 29517 "rand_subsamp_replay15"
jobs
pkill -f "rand_subsamp_replay15"
jobs

# # RAND SUB_SAMP + IN-DOMAIN PROMPT DATA (maybe with expbow)
# defaults

# export CUDA_VISIBLE_DEVICES=5
# # noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
# nohup sh script/updateapi.sh "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "rand_subsamp_indomain" 5002 & 
# # Other commands
# export CUDA_VISIBLE_DEVICES=6,7
# sh script/dpoplus_script.sh "bagofwords" "outputs/data/bagofwords/bowsynth50knozeros" "http://127.0.0.1:5002/train" 29517 "rand_subsamp_indomain"
# jobs
# pkill -f "rand_subsamp_indomain"
# jobs
