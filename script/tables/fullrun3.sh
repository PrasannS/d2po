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


# RANDOM SUBSAMPLING RUN from base (not using DPO)
defaults
export BASEMODEL="facebook/opt-125m"

export CUDA_VISIBLE_DEVICES=2
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/updateapi.sh "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "rand_subsamp_2obase" 5004 & 
# Other commands
export CUDA_VISIBLE_DEVICES=5,6
sh script/dpoplus_script.sh "bagofwords" "ultra" "http://127.0.0.1:5004/train" 29519 "rand_subsamp_2obase"
jobs
pkill -f "rand_subsamp_2obase"
jobs


# RAND SUBSAMPLE WITH LOWER LR
defaults
export ULR=3e-5

export CUDA_VISIBLE_DEVICES=2
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/updateapi.sh "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "rand_subsamp_2lowlr" 5004 & 
# Other commands
export CUDA_VISIBLE_DEVICES=5,6
sh script/dpoplus_script.sh "bagofwords" "ultra" "http://127.0.0.1:5004/train" 29519 "rand_subsamp_2lowlr"
jobs
pkill -f "rand_subsamp_2lowlr"
jobs

# RAND SUB_SAMP + IN-DOMAIN PROMPT DATA (maybe with expbow)
defaults

export CUDA_VISIBLE_DEVICES=2
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/updateapi.sh "bagofwords" "bowsynth50knozeros" "expbow50" "rand_subsamp_2indomain" 5004 & 
# Other commands
export CUDA_VISIBLE_DEVICES=5,6
sh script/dpoplus_script.sh "bagofwords" "outputs/data/bagofwords/bowsynth50knozeros" "http://127.0.0.1:5004/train" 29519 "rand_subsamp_2indomain"
jobs
pkill -f "rand_subsamp_2indomain"
jobs
