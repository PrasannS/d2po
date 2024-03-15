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
export RELABELS=$((8))
export CUDA_VISIBLE_DEVICES=5
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/newupdateapi.sh "math" "" "tiny_rm" "conf8repeasy3" 5001 & 
# Other command
export CUDA_VISIBLE_DEVICES=6,7
sh script/dpoplus_script.sh "math" "outputs/data/math/matheasier3" "http://127.0.0.1:5001/train" 29510 "conf8repeasy3"
jobs
pkill -f "conf8repeasy3"
jobs

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
export RELABELS=$((8))
export CUDA_VISIBLE_DEVICES=5
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/newupdateapi.sh "math" "" "tiny_rm" "conf8repeasy4" 5001 & 
# Other command
export CUDA_VISIBLE_DEVICES=6,7
sh script/dpoplus_script.sh "math" "outputs/data/math/matheasy4" "http://127.0.0.1:5001/train" 29510 "conf8repeasy4"
jobs
pkill -f "conf8repeasy4"
jobs