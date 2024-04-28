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
export DPOBASEAPI="outputs/models/math/mathbigdata1b"
# # noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/newupdateapi.sh "math" "" "mathbigdata1b" "1bbig_128_8_seed3_dpover" 5005 & 
# # Other commands
export CUDA_VISIBLE_DEVICES=6,7
sh script/dpoplus_script.sh "math" "outputs/data/math/mathppoinps200k" "http://127.0.0.1:5005/train" 29522 "1bbig_128_8_seed3_dpover"
jobs
pkill -f "1bbig_128_8_seed3_dpover"
jobs

export MLEN=50

export STEPS=1000

export PPOUPDATES=4
export DPOBATCHSIZE=8
export MBSIZE=8
export GBSIZE=8
export DPOBASEAPI="facebook/opt-125m"
export GREWARD="unique_nns"
export UEPOCHS=8
export BASEMODEL="facebook/opt-125m"
export SAMPN=$((60))
export RELABELS=$((5))
export CUDA_VISIBLE_DEVICES=5
nohup sh script/newupdateapi.sh "unique_nns" "" "newtiny_rm" "noun60_5_seed1dpover" 5012 & 
# # Other commands
export CUDA_VISIBLE_DEVICES=6,7
sh script/dpoplus_script.sh "unique_nns" "ultra" "http://127.0.0.1:5012/train" 29548 "noun60_5_seed1dpover"
jobs
pkill -f "noun60_5_seed1dpover"
jobs