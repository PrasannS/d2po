
# SH File for running the whole shebang, multiple jobs

export CFG=src/configs/ppo_2gpu.yaml
export STEPS=2000
export SUPDATES=10000000
export SEED=2
export MLEN=50
export KEEPLONG=0

defaults() {
    export BASEMODEL="outputs/models/bagofwords/bowtiny_dpo"
    export ATYPE="rand"
    export ULR=1e-4
    export CRATIO=1
}

# BOW examine behavior at smaller intervals
defaults
export SEED=2

export ATYPE="conf"
export UEPOCHS=1
export APBSIZE=16
export GREWARD="bagofwords"
# export DPOBASEAPI="facebook/opt-125m"

export DPOBATCHSIZE=32
export MBSIZE=32
export GBSIZE=32
export USEDPO=0
export ONLYOLDUPDATES=1

export SAMPN=$((32*5))
export RELABELS=$((1*5))
export CUDA_VISIBLE_DEVICES=3
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/newupdateapi.sh "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "conf_newalgo_5testnewseed2_oldupdates" 5004 & 
# Other commands
export CUDA_VISIBLE_DEVICES=4,5
sh script/dpoplus_script.sh "bagofwords" "ultra" "http://127.0.0.1:5004/train" 29527 "conf_newalgo_5testnewseed2_oldupdates"
jobs
pkill -f "conf_newalgo_5testnewseed2_oldupdates"
jobs

# Noun pick up the missing conf baseline
defaults
export BASEMODEL="outputs/models/nouns/smalldpo"

export ATYPE="conf"
export UEPOCHS=1
export APBSIZE=16
export SEED=2

export GREWARD="nouns"

export DPOBATCHSIZE=32
export MBSIZE=32
export GBSIZE=32

export SAMPN=$((32*5))
export RELABELS=$((2))
export CUDA_VISIBLE_DEVICES=3
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/newupdateapi.sh "nouns" "dponounsynth" "tiny_rm" "confnoun_newalgo_2_5_seed2_oldupdates" 5004 & 
sleep 20
# Other commands
export CUDA_VISIBLE_DEVICES=6,7
sh script/dpoplus_script.sh "nouns" "ultra" "http://127.0.0.1:5004/train" 29527 "confnoun_newalgo_2_5_seed2_oldupdates"
jobs
pkill -f "confnoun_newalgo_2_5_seed2_oldupdates"
jobs

# CDIST GET A BASELINE AT SMALLER INTERV LEVEL
defaults
export KEEPLONG=10

export ATYPE="rand"
export UEPOCHS=1
export APBSIZE=16
export GREWARD="contrastivedistill"
export BASEMODEL="outputs/models/contrastivedistill/smalldpo"


export DPOBATCHSIZE=32
export MBSIZE=32
export GBSIZE=32

export SAMPN=$((32*10))
export RELABELS=$((5))
export CUDA_VISIBLE_DEVICES=3
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/newupdateapi.sh "contrastivedistill" "contoptprefs" "tiny_rm" "rand_cdist_10_5_activefixseed2_oldupdates" 5004 & 
sleep 20
# Other commands
export CUDA_VISIBLE_DEVICES=6,7
sh script/dpoplus_script.sh "contrastivedistill" "outputs/data/contrastivedistill/wikionpprompts200k" "http://127.0.0.1:5004/train" 29527 "rand_cdist_10_5_activefixseed2_oldupdates"
jobs
pkill -f "rand_cdist_10_5_activefixseed2_oldupdates"
jobs