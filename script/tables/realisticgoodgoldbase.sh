# OFF POLICY VERSIONS OF ALL THE ACTIVE THINGS: 
# IMPORTANT BASELINE, WHAT HAPPENS WITH MORE DPO UPDATES ON THE SAME THING
export CFG=src/configs/ppo_2gpu.yaml
export STEPS=2000
export SUPDATES=10000000
export SEED=0
export KEEPLONG=10

# CONTRASTIVE DISTILL
defaults() {
    export BASEMODEL="outputs/models/ultra/tiny_dpo_tulu"
    export MLEN=256
    export ATYPE="conf"
    export ULR=1e-4
    export CRATIO=1
}

export USEDPO=0
export ONLYOLDUPDATES=0
export PPOUPDATES=2
defaults
export DPOBATCHSIZE=8
export MBSIZE=8
export GBSIZE=8

# # Other commands
# export CUDA_VISIBLE_DEVICES=0,1
# sh script/ipoplus_script.sh "contrastivedistill" "outputs/data/contrastivedistill/wikionpprompts200k" "functioncontrastivedistill" 29523 "contdistb8_moreupdates_4ups_ipo"
# jobs
# # pkill -f "justoffpolicy_conf_cdist_100_50_activefix"
# jobs

defaults
export BASEMODEL="outputs/models/ultra/tiny_dpo_tulu"
export DPOBATCHSIZE=8
export MBSIZE=2
export GBSIZE=8
export SEED=1
export RFREQ=0
export RUPS=0
export RBS=0

defaults() {
    export BASEMODEL="outputs/models/ultra/tiny_dpo_tulu"
    export ATYPE="rand"
    export ULR=1e-4
    export CRATIO=1
}

# BOW examine behavior at smaller intervals
defaults
export SEED=0

export ATYPE="conf"
export UEPOCHS=3
export APBSIZE=2
export GREWARD="eurusrm"

export DPOBATCHSIZE=32
export MBSIZE=2
export GBSIZE=8
export USEDPO=0
export PPOUPDATES=2
export ONLYOLDUPDATES=0

export SAMPN=$((32))
export RELABELS=$((2))
export SFREQ=5
export CUDA_VISIBLE_DEVICES=0
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/newupdateapi.sh "eurusrm" "" "eurusrm_fresh" "eurusrmbaseline32save2" 5004 & 
# # # Other commands
export CUDA_VISIBLE_DEVICES=1,2
sh script/dpoplus_replay.sh "eurusrm" "ultra" "http://127.0.0.1:5004/predeurus" 29528 "eurusrmbaseline32save2"

jobs
pkill -f "eurusrmbaseline32save2"
jobs