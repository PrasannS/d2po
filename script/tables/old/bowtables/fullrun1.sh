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

# NOTE SMALL CHANCE OF RACE CONDITIONS, WATCH OUT

# CONF SUB_SAMPLING

defaults
export ATYPE="conf"

export CUDA_VISIBLE_DEVICES=2
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/updateapi.sh "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "conf_subsamp" 5001 & 
# Other commands
export CUDA_VISIBLE_DEVICES=3,4
sh script/dpoplus_script.sh "bagofwords" "ultra" "http://127.0.0.1:5001/train" 29516 "conf_subsamp"
jobs
pkill -f "conf_subsamp"
jobs


# RANDOM SUBSAMPLING RUN 
defaults
export CUDA_VISIBLE_DEVICES=2
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/updateapi.sh "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "rand_subsamp_only" 5001 & 
# Other commands
export CUDA_VISIBLE_DEVICES=3,4
sh script/dpoplus_script.sh "bagofwords" "ultra" "http://127.0.0.1:5001/train" 29516 "rand_subsamp_only"
jobs
pkill -f "rand_subsamp_only"
jobs

# # RANDOM SUBSAMPLING RUN from base (not using DPO)
# defaults
# export BASEMODEL="facebook/opt-125m"

# export CUDA_VISIBLE_DEVICES=2
# # noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
# nohup sh script/updateapi.sh "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "rand_subsamp_obase" 5001 & 
# # Other commands
# export CUDA_VISIBLE_DEVICES=3,4
# sh script/dpoplus_script.sh "bagofwords" "ultra" "http://127.0.0.1:5001/train" 29516 "rand_subsamp_obase"
# jobs
# pkill -f "rand_subsamp_obase"
# jobs


# # RAND SUBSAMPLE WITH LOWER LR
# defaults
# export ULR=3e-5

# export CUDA_VISIBLE_DEVICES=2
# # noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
# nohup sh script/updateapi.sh "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "rand_subsamp_lowlr" 5001 & 
# # Other commands
# export CUDA_VISIBLE_DEVICES=3,4
# sh script/dpoplus_script.sh "bagofwords" "ultra" "http://127.0.0.1:5001/train" 29516 "rand_subsamp_lowlr"
# jobs
# pkill -f "rand_subsamp_lowlr"
# jobs


