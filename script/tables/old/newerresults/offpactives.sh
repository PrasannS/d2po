# OFF POLICY VERSIONS OF ALL THE ACTIVE THINGS: 

export CFG=src/configs/ppo_2gpu.yaml
export STEPS=2000
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

# BEST TECHNIQUE (ALL IN ONE)
defaults
export ATYPE="conf"
export UEPOCHS=3
export APBSIZE=16
export GREWARD="contrastivedistill"

export DPOBATCHSIZE=32
export MBSIZE=32
export GBSIZE=32

export SAMPN=$((32*100))
export RELABELS=$((50))
export CUDA_VISIBLE_DEVICES=2
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/newupdateapi.sh "contrastivedistill" "" "tiny_rm" "justoffpolicy_conf_cdist_100_50_activefix" 5010 & 
sleep 20
# Other commands
export CUDA_VISIBLE_DEVICES=3,4
sh script/dpoplus_script.sh "contrastivedistill" "outputs/data/contrastivedistill/wikionpprompts200k" "http://127.0.0.1:5010/train" 29519 "justoffpolicy_conf_cdist_100_50_activefix"
jobs
pkill -f "justoffpolicy_conf_cdist_100_50_activefix"
jobs

# NOUNS
defaults
export BASEMODEL="outputs/models/nouns/smalldpo"

export ATYPE="conf"
export UEPOCHS=3
export APBSIZE=16
export GREWARD="nouns"

export DPOBATCHSIZE=32
export MBSIZE=32
export GBSIZE=32

export SAMPN=$((32*50))
export RELABELS=$((50))
export CUDA_VISIBLE_DEVICES=2
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/newupdateapi.sh "nouns" "" "tiny_rm" "justoffpolicy_confnoun_newalgo_100small_200testv3" 5010 & 
sleep 20
# Other commands
export CUDA_VISIBLE_DEVICES=3,4
sh script/dpoplus_script.sh "nouns" "ultra" "http://127.0.0.1:5010/train" 29519 "justoffpolicy_confnoun_newalgo_100small_200testv3"
jobs
pkill -f "justoffpolicy_confnoun_newalgo_100small_200testv3"
jobs

defaults

export BASEMODEL="outputs/models/bagofwords/bowtiny_dpo"
export ATYPE="conf"
export UEPOCHS=3
export APBSIZE=16
export GREWARD="bagofwords"

export DPOBATCHSIZE=32
export MBSIZE=32
export GBSIZE=32

export SAMPN=$((32*500))
export RELABELS=$((1*500))
export CUDA_VISIBLE_DEVICES=2
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/newupdateapi.sh "bagofwords" "" "bowtiny_rm" "justoffpolicy_conf_newalgo_500" 5010 & 
# Other commands
export CUDA_VISIBLE_DEVICES=3,4
sh script/dpoplus_script.sh "bagofwords" "ultra" "http://127.0.0.1:5010/train" 29519 "justoffpolicy_conf_newalgo_500"
jobs
pkill -f "justoffpolicy_conf_newalgo_500"
jobs
