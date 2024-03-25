

# SH File for running the whole shebang, multiple jobs

export CFG=src/configs/ppo_2gpu.yaml
export STEPS=2000
export SUPDATES=10000000
export SEED=2
export KEEPLONG=0
export MLEN=50
export PPOUPDATES=1
defaults() {
    export BASEMODEL="outputs/models/bagofwords/bowtiny_dpo"
    export ATYPE="rand"
    export ULR=1e-4
    export CRATIO=1
}

export USEDPO=0
export ONLYOLDUPDATES=0

# BOW examine behavior at smaller intervals
defaults
export SEED=2

export ATYPE="conf"
export UEPOCHS=3
export APBSIZE=16
export GREWARD="bagofwords"

export DPOBATCHSIZE=32
export MBSIZE=32
export GBSIZE=32

export SAMPN=$((32*5))
export RELABELS=$((1*5))
export CUDA_VISIBLE_DEVICES=0
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/newupdateapi.sh "bagofwords" "" "bowtiny_rm" "conf_newalgo_5testnormppov2" 5003 & 
# Other commands
export CUDA_VISIBLE_DEVICES=1,2
sh script/dpoplus_script.sh "bagofwords" "ultra" "http://127.0.0.1:5003/train" 29520 "conf_newalgo_5testnormppov2"
jobs
pkill -f "conf_newalgo_5testnormppov2"
jobs

# # Noun pick up the missing conf baseline


# export CFG=src/configs/ppo_2gpu.yaml
# export SUPDATES=10000000
# export SEED=2
# export KEEPLONG=10

# # CDIST GET A BASELINE AT SMALLER INTERV LEVEL
# defaults
# export ATYPE="rand"
# export UEPOCHS=3
# export APBSIZE=16
# export GREWARD="contrastivedistill"
# export BASEMODEL="outputs/models/contrastivedistill/smalldpo"

# export DPOBATCHSIZE=32
# export MBSIZE=32
# export GBSIZE=32

# export SAMPN=$((32*10))
# export RELABELS=$((5))
# export CUDA_VISIBLE_DEVICES=0
# # noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
# nohup sh script/newupdateapi.sh "contrastivedistill" "" "tiny_rm" "rand_cdist_10_5_normppo" 5010 & 
# sleep 20
# # Other commands
# export CUDA_VISIBLE_DEVICES=3,4
# sh script/dpoplus_script.sh "contrastivedistill" "outputs/data/contrastivedistill/wikionpprompts200k" "http://127.0.0.1:5010/train" 29519 "rand_cdist_10_5_normppo"
# jobs
# pkill -f "rand_cdist_10_5_normppo"
# jobs

# defaults
# export BASEMODEL="outputs/models/nouns/smalldpo"

# export ATYPE="conf"
# export UEPOCHS=3
# export APBSIZE=16
# export SEED=2

# export GREWARD="nouns"

# export DPOBATCHSIZE=32
# export MBSIZE=32
# export GBSIZE=32

# export SAMPN=$((32*5))
# export RELABELS=$((2))
# export CUDA_VISIBLE_DEVICES=0
# # noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
# nohup sh script/newupdateapi.sh "nouns" "" "tiny_rm" "confnoun_newalgo_2_5_normppo" 5010 & 
# sleep 20
# # Other commands
# export CUDA_VISIBLE_DEVICES=3,4
# sh script/dpoplus_script.sh "nouns" "ultra" "http://127.0.0.1:5010/train" 29519 "confnoun_newalgo_2_5_normppo"
# jobs
# pkill -f "confnoun_newalgo_2_5_normppo"
# jobs

# SH File for running the whole shebang on cdistill, noun tasks, will debug math separately