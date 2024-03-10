export CFG=src/configs/ppo_2gpu.yaml
export STEPS=200
export LABTHRESH=0.3
export SUPDATES=10000000
export SEED=0
export KEEPLONG=0
export MLEN=50

defaults() {
    export REDOBATCH=1
    export LABRATIO=0.25
    export ATYPE="rand"
    export ULR=1e-4
    export DPOBATCHSIZE=32
}

# NOTE SMALL CHANCE OF RACE CONDITIONS, WATCH OUT

# CONF SUB_SAMPLING

defaults

export BASEMODEL="outputs/models/ultra/tiny_dpo_tulu"
export MLEN=256
# BATCH SIZE OF 8 instead of 32 (make gold set equivalent basically), how far can we get
export CUDA_VISIBLE_DEVICES=1,2

export DPOBATCHSIZE=4
export MBSIZE=4
sh script/dpoplus_script.sh "ultra" "outputs/data/ultra/smallultrappoinps" "functionbagofwords" 29520 "ultragolddpoplus"

# sh script/dpoplus_script.sh "ultra" "ultra" "functionultrafeedbackgold" 29520 "ultragolddpoplus"
