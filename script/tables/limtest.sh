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

# RANDOM SUBSAMPLING RUN MATH
defaults

export LABRATIO=0.0625
export ATYPE="conf"
export BASEMODEL="outputs/models/math/smalldpo"
export DPOBATCHSIZE=128
export MLEN=100
export APBSIZE=32

export CUDA_VISIBLE_DEVICES=3
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/updateapi.sh "math" "" "tiny_rm" "conf_subsamp_bigrolloutmath" 5004 & 
sleep 10
# Other commands
export CUDA_VISIBLE_DEVICES=4,5
sh script/dpoplus_script.sh "math" "outputs/data/math/mathppoinps" "http://127.0.0.1:5004/train" 29519 "conf_subsamp_bigrolloutmath"
jobs
pkill -f "conf_subsamp_bigrolloutmath"
jobs

export LABRATIO=0.0625
export ATYPE="conf"
export BASEMODEL="outputs/models/contrastivedistill/smalldpo"
export DPOBATCHSIZE=128
export MLEN=50
export APBSIZE=32

export CUDA_VISIBLE_DEVICES=3
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/updateapi.sh "contrastivedistill" "" "tiny_rm" "conf_subsamp_bigrolloutcdist" 5004 & 
# Other commands
export CUDA_VISIBLE_DEVICES=4,5
sh script/dpoplus_script.sh "contrastivedistill" "outputs/data/contrastivedistill/wikionpolicyprompts" "http://127.0.0.1:5004/train" 29522 "conf_subsamp_bigrolloutcdist"
jobs
pkill -f "conf_subsamp_bigrolloutcdist"
jobs