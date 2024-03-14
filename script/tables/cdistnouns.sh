# SH File for running the whole shebang on cdistill, noun tasks, will debug math separately

export CFG=src/configs/ppo_2gpu.yaml
export STEPS=2000
export SUPDATES=10000000
export SEED=0
export KEEPLONG=0

defaults() {
    export BASEMODEL="outputs/models/contrastivedistill/smalldpo"
    export MLEN=50
    export ATYPE="conf"
    export ULR=1e-4
    export CRATIO=1
}


# # BEST TECHNIQUE (ALL IN ONE)
# defaults
# export ATYPE="conf"
# export UEPOCHS=3
# export APBSIZE=16
# export GREWARD="contrastivedistill"

# export DPOBATCHSIZE=32
# export MBSIZE=32
# export GBSIZE=32

# export SAMPN=$((32*500))
# export RELABELS=$((250))
# export CUDA_VISIBLE_DEVICES=1
# # noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
# nohup sh script/newupdateapi.sh "contrastivedistill" "" "tiny_rm" "conf_cdist_500_250_active3" 5009 & 
# sleep 20
# # Other commands
# export CUDA_VISIBLE_DEVICES=0,4
# sh script/dpoplus_script.sh "contrastivedistill" "outputs/data/contrastivedistill/wikionpprompts200k" "http://127.0.0.1:5009/train" 29520 "conf_cdist_500_250_active3"
# jobs
# pkill -f "conf_cdist_500_250_active3"
# jobs

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
export RELABELS=$((25))
export CUDA_VISIBLE_DEVICES=1
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/newupdateapi.sh "nouns" "" "tiny_rm" "confnoun_newalgo_100small_200test" 5009 & 
sleep 20
# Other commands
export CUDA_VISIBLE_DEVICES=0,4
sh script/dpoplus_script.sh "nouns" "ultra" "http://127.0.0.1:5009/train" 29520 "confnoun_newalgo_100small_200test"
jobs
pkill -f "confnoun_newalgo_100small_200test"
jobs

# NOUNS
defaults
export BASEMODEL="outputs/models/nouns/smalldpo"

export ATYPE="rand"
export UEPOCHS=3
export APBSIZE=16
export GREWARD="nouns"

export DPOBATCHSIZE=32
export MBSIZE=32
export GBSIZE=32

export SAMPN=$((32*50))
export RELABELS=$((25))
export CUDA_VISIBLE_DEVICES=1
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/newupdateapi.sh "nouns" "" "tiny_rm" "randnoun_newalgo_100small_200test" 5009 & 
sleep 20
# Other commands
export CUDA_VISIBLE_DEVICES=0,4
sh script/dpoplus_script.sh "nouns" "ultra" "http://127.0.0.1:5009/train" 29520 "randnoun_newalgo_100small_200test"
jobs
pkill -f "randnoun_newalgo_100small_200test"
jobs

# # BEST TECHNIQUE (ALL IN ONE)
# defaults
# export ATYPE="random"
# export UEPOCHS=3
# export APBSIZE=16
# export GREWARD="contrastivedistill"

# export DPOBATCHSIZE=32
# export MBSIZE=32
# export GBSIZE=32

# export SAMPN=$((32*500))
# export RELABELS=$((250))
# export CUDA_VISIBLE_DEVICES=1
# # noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
# nohup sh script/newupdateapi.sh "contrastivedistill" "" "tiny_rm" "rand_cdist_500_250_active3" 5009 & 
# sleep 20
# # Other commands
# export CUDA_VISIBLE_DEVICES=0,4
# sh script/dpoplus_script.sh "contrastivedistill" "outputs/data/contrastivedistill/wikionpprompts200k" "http://127.0.0.1:5009/train" 29520 "rand_cdist_500_250_active3"
# jobs
# pkill -f "rand_cdist_500_250_active3"
# jobs