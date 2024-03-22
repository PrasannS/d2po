
# SH File for running the whole shebang, multiple jobs

export CFG=src/configs/ppo_2gpu.yaml
export STEPS=2000
export SUPDATES=10000000
export SEED=0
export KEEPLONG=0
export MLEN=50

defaults() {
    export BASEMODEL="outputs/models/bagofwords/bowtiny_dpo"
    export ATYPE="rand"
    export ULR=1e-4
    export CRATIO=1
}

# BOW examine behavior at smaller intervals
defaults
export ATYPE="conf"
export UEPOCHS=2
export APBSIZE=16
export GREWARD="bagofwords"

export DPOBATCHSIZE=32
export MBSIZE=32
export GBSIZE=32

export SAMPN=$((32*5))
export RELABELS=$((1*5))
export CUDA_VISIBLE_DEVICES=0
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/newupdateapi.sh "bagofwords" "" "bowtiny_rm" "conf_newalgo_5testnew" 5003 & 
# Other commands
export CUDA_VISIBLE_DEVICES=1,2
sh script/dpoplus_script.sh "bagofwords" "ultra" "http://127.0.0.1:5003/train" 29519 "conf_newalgo_5testnew"
jobs
pkill -f "conf_newalgo_5testnew"
jobs

# Noun pick up the missing conf baseline
defaults
export BASEMODEL="outputs/models/nouns/smalldpo"

export ATYPE="conf"
export UEPOCHS=3
export APBSIZE=16
export GREWARD="nouns"

export DPOBATCHSIZE=32
export MBSIZE=32
export GBSIZE=32

export SAMPN=$((32*5))
export RELABELS=$((5))
export CUDA_VISIBLE_DEVICES=0
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/newupdateapi.sh "nouns" "" "tiny_rm" "confnoun_newalgo_5small_5testv3new" 5003 & 
sleep 20
# Other commands
export CUDA_VISIBLE_DEVICES=1,2
sh script/dpoplus_script.sh "nouns" "ultra" "http://127.0.0.1:5003/train" 29519 "confnoun_newalgo_5small_5testv3new"
jobs
pkill -f "confnoun_newalgo_5small_5testv3new"
jobs

# Noun pick up the missing conf baseline
defaults
export BASEMODEL="outputs/models/nouns/smalldpo"

export ATYPE="conf"
export UEPOCHS=3
export APBSIZE=16
export GREWARD="nouns"

export DPOBATCHSIZE=32
export MBSIZE=32
export GBSIZE=32

export SAMPN=$((32*5))
export RELABELS=$((2))
export CUDA_VISIBLE_DEVICES=0
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/newupdateapi.sh "nouns" "" "tiny_rm" "confnoun_newalgo_2small_5testv3new" 5003 & 
sleep 20
# Other commands
export CUDA_VISIBLE_DEVICES=1,2
sh script/dpoplus_script.sh "nouns" "ultra" "http://127.0.0.1:5003/train" 29519 "confnoun_newalgo_2small_5testv3new"
jobs
pkill -f "confnoun_newalgo_2small_5testv3new"
jobs