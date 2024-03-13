# SH File for running the whole shebang, multiple jobs

export CFG=src/configs/ppo_2gpu.yaml
export STEPS=2000
export SUPDATES=10000000
export SEED=0
export KEEPLONG=0

defaults() {
    export BASEMODEL="outputs/models/math/smalldpo"
    export MLEN=100
    export ATYPE="rand"
    export ULR=1e-4
    export CRATIO=1
}

# BEST TECHNIQUE (ALL IN ONE)
defaults
export ATYPE="conf"
export UEPOCHS=3
export APBSIZE=16
export GREWARD="math"

export DPOBATCHSIZE=32
export MBSIZE=32
export GBSIZE=32

export SAMPN=$((32*1))
export RELABELS=$((4*1))
export CUDA_VISIBLE_DEVICES=7
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/newupdateapi.sh "math" "" "tiny_rm" "conf_newalgo_1_4test" 5001 & 
# Other commands
export CUDA_VISIBLE_DEVICES=2,3
sh script/dpoplus_script.sh "math" "outputs/data/math/mathppoinps" "http://127.0.0.1:5001/train" 29510 "conf_newalgo_1_4test"
jobs
pkill -f "conf_newalgo_1_4test"
jobs

defaults
export ATYPE="conf"
export UEPOCHS=3
export APBSIZE=16
export GREWARD="math"

export DPOBATCHSIZE=32
export MBSIZE=32
export GBSIZE=32

export SAMPN=$((32*100))
export RELABELS=$((4*100))
export CUDA_VISIBLE_DEVICES=7
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/newupdateapi.sh "math" "" "tiny_rm" "conf_newalgo_100_4test" 5001 & 
# Other commands
export CUDA_VISIBLE_DEVICES=2,3
sh script/dpoplus_script.sh "math" "outputs/data/math/mathppoinps" "http://127.0.0.1:5001/train" 29510 "conf_newalgo_100_4test"
jobs
pkill -f "conf_newalgo_100_4test"
jobs