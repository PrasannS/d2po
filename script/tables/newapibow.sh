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

# BEST TECHNIQUE (ALL IN ONE)
defaults
export ATYPE="conf"
export UEPOCHS=3
export APBSIZE=16
export GREWARD="bagofwords"

export DPOBATCHSIZE=32
export MBSIZE=32
export GBSIZE=32


export SAMPN=$((32*1))
export RELABELS=$((1*1))
export CUDA_VISIBLE_DEVICES=4
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/newupdateapi.sh "bagofwords" "" "bowtiny_rm" "conf_newalgo_1test" 5005 & 
# Other commands
export CUDA_VISIBLE_DEVICES=5,6
sh script/dpoplus_script.sh "bagofwords" "ultra" "http://127.0.0.1:5005/train" 29518 "conf_newalgo_1test"
jobs
pkill -f "conf_newalgo_1test"
jobs

defaults

export SAMPN=$((32*1))
export RELABELS=$((2*1))
export CUDA_VISIBLE_DEVICES=4
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/newupdateapi.sh "bagofwords" "" "bowtiny_rm" "conf_newalgo_4test" 5005 & 
# Other commands
export CUDA_VISIBLE_DEVICES=5,6
sh script/dpoplus_script.sh "bagofwords" "ultra" "http://127.0.0.1:5005/train" 29518 "conf_newalgo_4test"
jobs
pkill -f "conf_newalgo_4test"
jobs

defaults

export SAMPN=$((32*25))
export RELABELS=$((2*25))
export CUDA_VISIBLE_DEVICES=4
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/newupdateapi.sh "bagofwords" "" "bowtiny_rm" "conf_newalgo_2test25" 5005 & 
# Other commands
export CUDA_VISIBLE_DEVICES=5,6
sh script/dpoplus_script.sh "bagofwords" "ultra" "http://127.0.0.1:5005/train" 29518 "conf_newalgo_2test25"
jobs
pkill -f "conf_newalgo_2test25"
jobs

defaults