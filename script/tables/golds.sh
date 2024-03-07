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

# BATCH SIZE OF 8 instead of 32 (make gold set equivalent basically), how far can we get
export CUDA_VISIBLE_DEVICES=1,7
sh script/dpoplus_cbsize.sh "bagofwords" "ultra" "functionbagofwords" 29520 "goldb8_dpobase_v2"

export BASEMODEL="facebook/opt-125m"
sh script/dpoplus_cbsize.sh "bagofwords" "ultra" "functionbagofwords" 29520 "goldb8_optbase"
