# SH File for running the whole shebang, multiple jobs

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
    export ULR=1e-4
}


# RANDOM SUBSAMPLING RUN MATH
defaults

export BASEMODEL="outputs/models/math/smalldpo"
export MLEN=100

export CUDA_VISIBLE_DEVICES=3
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/updateapi.sh "math" "" "tiny_rm" "rand_subsamp_onlymath" 5004 & 
# Other commands
export CUDA_VISIBLE_DEVICES=4,5
sh script/dpoplus_script.sh "math" "outputs/data/math/mathppoinps" "http://127.0.0.1:5004/train" 29519 "rand_subsamp_onlymath"
jobs
pkill -f "rand_subsamp_onlymath"
jobs

export ATYPE="conf"
# CONF SUBSAMPLING RUN MATH
export CUDA_VISIBLE_DEVICES=3
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/updateapi.sh "math" "" "tiny_rm" "conf_subsamp_onlymath" 5004 & 
# Other commands
export CUDA_VISIBLE_DEVICES=4,5
sh script/dpoplus_script.sh "math" "outputs/data/math/mathppoinps" "http://127.0.0.1:5004/train" 29519 "conf_subsamp_onlymath"
jobs
pkill -f "conf_subsamp_onlymath"
jobs

# NOUN setting with random sampling
export BASEMODEL="outputs/models/nouns/smalldpo"
export MLEN=50
export ATYPE="rand"


export CUDA_VISIBLE_DEVICES=3
nohup sh script/updateapi.sh "nouns" "" "tiny_rm" "rand_subsamp_onlynouns" 5004 & 
# Other commands
export CUDA_VISIBLE_DEVICES=4,5
sh script/dpoplus_script.sh "nouns" "ultra" "http://127.0.0.1:5004/train" 29519 "rand_subsamp_onlynouns"
jobs
pkill -f "rand_subsamp_onlynouns"
jobs
