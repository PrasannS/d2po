
export CFG=src/configs/ppo_2gpu.yaml
export STEPS=3000
export SUPDATES=10000000
export SEED=0
export KEEPLONG=0
export MLEN=50

defaults() {
    export BASEMODEL="outputs/models/bagofwords/bowtiny_dpo"
    export ATYPE="conf"
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

export SAMPN=$((32*500))
export RELABELS=$((100))
export CUDA_VISIBLE_DEVICES=1
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/newupdateapi.sh "bagofwords" "" "bowtiny_rm" "conf_newalgo_100small_500test" 5003 & 
# Other commands
export CUDA_VISIBLE_DEVICES=2,3
sh script/dpoplus_script.sh "bagofwords" "ultra" "http://127.0.0.1:5003/train" 29519 "conf_newalgo_100small_500test"
jobs
pkill -f "conf_newalgo_100small_500test"
jobs

# BEST TECHNIQUE (ALL IN ONE)
defaults
export ATYPE="conf"
export UEPOCHS=3
export APBSIZE=16
export GREWARD="bagofwords"

export DPOBATCHSIZE=32
export MBSIZE=32
export GBSIZE=32

export SAMPN=$((32*1000))
export RELABELS=$((100))
export CUDA_VISIBLE_DEVICES=1
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/newupdateapi.sh "bagofwords" "" "bowtiny_rm" "conf_newalgo_100small_1000test" 5003 & 
# Other commands
export CUDA_VISIBLE_DEVICES=2,3
sh script/dpoplus_script.sh "bagofwords" "ultra" "http://127.0.0.1:5003/train" 29519 "conf_newalgo_100small_1000test"
jobs
pkill -f "conf_newalgo_100small_1000test"
jobs