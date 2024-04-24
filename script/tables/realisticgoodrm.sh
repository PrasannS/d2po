# SH File for running the whole shebang, fixed method up with more code

export CFG=src/configs/ppo_2gpu.yaml
export STEPS=2000
export SUPDATES=10000000
export SEED=0
export KEEPLONG=0
export MLEN=256

defaults() {
    export BASEMODEL="outputs/models/ultra/tiny_dpo_tulu"
    export ATYPE="rand"
    export ULR=1e-4
    export CRATIO=1
}

# BOW examine behavior at smaller intervals
defaults
export SEED=0

export ATYPE="conf"
export UEPOCHS=3
export APBSIZE=2
export GREWARD="eurusrm"

export DPOBATCHSIZE=32
export MBSIZE=2
export GBSIZE=8
export USEDPO=0
export PPOUPDATES=1
export ONLYOLDUPDATES=0

export SAMPN=$((32))
export RELABELS=$((4))
export CUDA_VISIBLE_DEVICES=4
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/newupdateapi.sh "eurusrm" "" "tiny_rm" "mainalgo_32_4justoffpolicy" 5003 & 
export SFREQ=25
# Other commands
export CUDA_VISIBLE_DEVICES=5,6
sh script/dpoplus_script.sh "eurusrm" "ultra" "http://127.0.0.1:5003/train" 29519 "mainalgo_32_4justoffpolicy"
jobs
pkill -f "mainalgo_32_4justoffpolicy"
jobs