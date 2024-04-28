# script for self-reward, DPO-based RM conifgs for main results

# SH File for running the whole shebang, fixed method up with more code

export CFG=src/configs/ppo_2gpu.yaml
export STEPS=500
export SUPDATES=10000000
export KEEPLONG=0
export MLEN=75
export ATYPE="rand"
export ULR=1e-4
export CRATIO=1

# BOW examine behavior at smaller intervals
export SEED=3
export ATYPE="conf"
export APBSIZE=16
export GREWARD="math"


export DPOBATCHSIZE=32
export MBSIZE=32
export GBSIZE=32
export PPOUPDATES=1
export ONLYOLDUPDATES=0

export SAMPN=$((256))
export RELABELS=$((16))
export BASEMODEL="outputs/models/math/mathbigdata1b"
export CUDA_VISIBLE_DEVICES=3

export UEPOCHS=5
export USEDPO=1
export SAMPN=$((128))
export RELABELS=$((8))
export CUDA_VISIBLE_DEVICES=5
export SFREQ=25
export PPOUPDATES=8
export OSAMP=4
export PPOUPDATES=8

# sh script/selfreward_script.sh "bagofwords" "ultra" "functionbagofwords" 29524 "confbow_goldb8_selfreward"

export SRROLLOUTS=$((32))
export SRSTEPS=$((4))

export DPOBASEAPI="outputs/models/math/mathbigdata1b"
# # noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
# # Other commands

export CUDA_VISIBLE_DEVICES=0,3
sh script/selfreward_script.sh "math" "outputs/data/math/mathppoinps200k" "functionmath" 29551 "mathselfrewtest"

export MLEN=50

export STEPS=1000

export PPOUPDATES=8
export DPOBATCHSIZE=32
export MBSIZE=32
export GBSIZE=32
export DPOBASEAPI="facebook/opt-125m"
export GREWARD="unique_nns"
export UEPOCHS=8
export BASEMODEL="facebook/opt-125m"
export SAMPN=$((60))
export RELABELS=$((5))
export CUDA_VISIBLE_DEVICES=5
# # Other commands
export CUDA_VISIBLE_DEVICES=0,3
sh script/selfreward_script.sh "unique_nns" "ultra" "functionunique_nns" 29551 "unn_selfrewtest"
