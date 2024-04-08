# OFF POLICY VERSIONS OF ALL THE ACTIVE THINGS: 
# IMPORTANT BASELINE, WHAT HAPPENS WITH MORE DPO UPDATES ON THE SAME THING
export CFG=src/configs/ppo_2gpu.yaml
export STEPS=2
export SUPDATES=10000000
export SEED=0
export KEEPLONG=10

# CONTRASTIVE DISTILL
defaults() {
    export BASEMODEL="outputs/models/contrastivedistill/smalldpo"
    export MLEN=50
    export ATYPE="conf"
    export ULR=1e-4
    export CRATIO=1
}

export USEDPO=0
export ONLYOLDUPDATES=0
export PPOUPDATES=1
defaults

export DPOBATCHSIZE=32
export MBSIZE=32
export GBSIZE=32
export STEPS=2000

# # Other commands
export CUDA_VISIBLE_DEVICES=1,2
sh script/dpoplus_script.sh "contrastivedistill" "outputs/data/contrastivedistill/wikionpprompts200k" "contdfixed" 29523 "dpoplusfullrm"
# jobs
# # pkill -f "justoffpolicy_conf_cdist_100_50_activefix"
# jobs

# # NOUNS ___________
# defaults
# export BASEMODEL="outputs/models/nouns/smalldpo"


# export DPOBATCHSIZE=32
# export MBSIZE=32
# export GBSIZE=32

# export STEPS=1000
# # # Other commands
# export CUDA_VISIBLE_DEVICES=1,2
# sh script/dpoplus_script.sh "nouns" "ultra" "dponounsynth_125magnfa" 29523 "normppofullrm"

export STEPS=2000

# # # #  __________________--
# defaults
# export BASEMODEL="outputs/models/bagofwords/bowtiny_dpo"


# export DPOBATCHSIZE=32
# export MBSIZE=32
# export GBSIZE=32

# # # # Other commands
# export CUDA_VISIBLE_DEVICES=1,2
# sh script/dpoplus_script.sh "bagofwords" "ultra" "expbow50" 29523 "normppofullrm"
