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

defaults

export BASEMODEL="outputs/models/math/smalldpo"
export MLEN=100
# BATCH SIZE OF 8 instead of 32 (make gold set equivalent basically), how far can we get
export CUDA_VISIBLE_DEVICES=0,1
export GBSIZE=8
export MBSIZE=8


export DPOBATCHSIZE=8
# earlier runs
# sh script/dpoplus_script.sh "math" "outputs/data/math/matheasier3" "functionmath" 29520 "goldb8easier3"

# sh script/dpoplus_script.sh "math" "outputs/data/math/matheasy4" "functionmath" 29520 "goldb8easy4"

# later runs
sh script/dpoplus_script.sh "math" "outputs/data/math/easy2_100k" "functionmath" 29527 "goldb8easy2fix"