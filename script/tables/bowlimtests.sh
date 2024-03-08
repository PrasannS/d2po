# SH File for running the whole shebang, multiple jobs
# test limit by pushing up PPO samples (maybe keep minibatch size)
export CFG=src/configs/ppo_2gpu.yaml
export STEPS=2000
export LABTHRESH=0.3
export SUPDATES=10000000
export SEED=0
export KEEPLONG=0
export MLEN=50

defaults() {
    export BASEMODEL="outputs/models/bagofwords/bowtiny_dpo"
    export REDOBATCH=5
    export LABRATIO=0.25
    export ATYPE="rand"
    export DPOBATCHSIZE=32
    export MBSIZE=32
    export ULR=1e-4
    export APBSIZE=16
    export CRATIO=1
}

defaults
# CONF SUBSAMPLING RUN BOW, 1/3 smaller label frequency
export CUDA_VISIBLE_DEVICES=3
export CRATIO=3

# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/updateapi.sh "bagofwords" "" "bowtiny_rm" "conf_subsamp_3intervrollout" 5005 & 
# Other commands
export CUDA_VISIBLE_DEVICES=6,7
sh script/dpoplus_script.sh "bagofwords" "ultra" "http://127.0.0.1:5005/train" 29520 "conf_subsamp_3intervrollout"
jobs
pkill -f "conf_subsamp_3intervrollout"
jobs

defaults

export LABRATIO=0.0625
export ATYPE="conf"
export DPOBATCHSIZE=128
export MLEN=100
export APBSIZE=32

# CONF SUBSAMPLING RUN BOW, gigantic batches
export CUDA_VISIBLE_DEVICES=3
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/updateapi.sh "bagofwords" "" "bowtiny_rm" "conf_subsamp_bigrolloutbow" 5005 & 
# Other commands
export CUDA_VISIBLE_DEVICES=6,7
sh script/dpoplus_script.sh "bagofwords" "ultra" "http://127.0.0.1:5005/train" 29520 "conf_subsamp_bigrolloutbow"
jobs
pkill -f "conf_subsamp_bigrolloutbow"
jobs


