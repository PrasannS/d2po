# SH File for running the whole shebang on cdistill, noun tasks, will debug math separately

export CFG=src/configs/ppo_2gpu.yaml
export STEPS=2000
export SUPDATES=10000000
export SEED=0
export KEEPLONG=10

defaults() {
    export BASEMODEL="outputs/models/contrastivedistill/smalldpo"
    export MLEN=50
    export ATYPE="conf"
    export ULR=1e-4
    export CRATIO=1
}

# BEST TECHNIQUE (ALL IN ONE)
defaults
export ATYPE="rand"
export UEPOCHS=3
export APBSIZE=16
export GREWARD="contrastivedistill"

export DPOBATCHSIZE=32
export MBSIZE=32
export GBSIZE=32

export SAMPN=$((32*100))
export RELABELS=$((50))
export CUDA_VISIBLE_DEVICES=5
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/newupdateapi.sh "contrastivedistill" "" "tiny_rm" "rand_cdist_100_50_activefix" 5010 & 
sleep 20
# Other commands
export CUDA_VISIBLE_DEVICES=2,3
sh script/dpoplus_script.sh "contrastivedistill" "outputs/data/contrastivedistill/wikionpprompts200k" "http://127.0.0.1:5010/train" 29522 "rand_cdist_100_50_activefix"
jobs
pkill -f "rand_cdist_100_50_activefix"
jobs

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
export CUDA_VISIBLE_DEVICES=5
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/newupdateapi.sh "contrastivedistill" "" "tiny_rm" "conf_cdist_100_50_activefix" 5010 & 
sleep 20
# Other commands
export CUDA_VISIBLE_DEVICES=2,3
sh script/dpoplus_script.sh "contrastivedistill" "outputs/data/contrastivedistill/wikionpprompts200k" "http://127.0.0.1:5010/train" 29522 "conf_cdist_100_50_activefix"
jobs
pkill -f "conf_cdist_100_50_activefix"
jobs


