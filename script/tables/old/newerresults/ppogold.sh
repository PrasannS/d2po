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
    export DPOBATCHSIZE=32
}

# NOTE SMALL CHANCE OF RACE CONDITIONS, WATCH OUT

# CONF SUB_SAMPLING

# defaults

# export MBSIZE=8
# export MLEN=50
# export BASEMODEL="outputs/models/contrastivedistill/smalldpo"
# # BATCH SIZE OF 8 instead of 32 (make gold set equivalent basically), how far can we get
# export CUDA_VISIBLE_DEVICES=0,1

# export DPOBATCHSIZE=8
# export GBSIZE=8
# sh script/dpoplus_script.sh "contrastivedistill" "outputs/data/contrastivedistill/wikionpprompts200k" "functioncontrastivedistill" 29520 "goldb8normppo"

# defaults

# export BASEMODEL="outputs/models/nouns/smalldpo"
# export DPOBATCHSIZE=8
# sh script/dpoplus_script.sh "nouns" "ultra" "functionnouns" 29520 "goldb8normppo"

defaults

export BASEMODEL="outputs/models/bagofwords/bowtiny_dpo"
export DPOBATCHSIZE=8
sh script/dpoplus_script.sh "bagofwords" "ultra" "functionbagofwords" 29520 "goldb8normppo"