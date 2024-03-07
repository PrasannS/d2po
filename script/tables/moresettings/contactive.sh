# SH File for running the whole shebang, multiple jobs

export CFG=src/configs/ppo_2gpu.yaml
export STEPS=2000
export LABTHRESH=0.3
export SUPDATES=10000000
export SEED=0
export KEEPLONG=0
export MLEN=50

defaults() {
    export BASEMODEL="outputs/models/contrastivedistill/bowtiny_dpo"
    export REDOBATCH=5
    export LABRATIO=0.25
    export ATYPE="rand"
    export DPOBATCHSIZE=32
    export ULR=1e-4
}

# RANDOM SUBSAMPLING RUN MATH
defaults

export BASEMODEL="outputs/models/contrastivedistill/smalldpo"
export MLEN=50

export CUDA_VISIBLE_DEVICES=3
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/updateapi.sh "contrastivedistill" "" "tiny_rm" "rand_subsamp_onlycdist" 5006 & 
sleep 10
# Other commands
export CUDA_VISIBLE_DEVICES=6,7
sh script/dpoplus_script.sh "contrastivedistill" "outputs/data/contrastivedistill/wikionpolicyprompts" "http://127.0.0.1:5006/train" 29522 "rand_subsamp_onlycdist"
jobs
pkill -f "rand_subsamp_onlycdist"
jobs

export ATYPE="conf"
# CONF SUBSAMPLING RUN MATH
export CUDA_VISIBLE_DEVICES=3
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/updateapi.sh "contrastivedistill" "" "tiny_rm" "conf_subsamp_onlycdist" 5006 & 
sleep 10
# Other commands
export CUDA_VISIBLE_DEVICES=6,7
sh script/dpoplus_script.sh "contrastivedistill" "outputs/data/contrastivedistill/wikionpolicyprompts" "http://127.0.0.1:5006/train" 29522 "conf_subsamp_onlycdist"
jobs
pkill -f "conf_subsamp_onlycdist"
jobs

# NOUN setting with random sampling
export BASEMODEL="outputs/models/nouns/smalldpo"
export MLEN=50
export ATYPE="conf"


export CUDA_VISIBLE_DEVICES=3
nohup sh script/updateapi.sh "nouns" "" "tiny_rm" "conf_subsamp_onlynouns" 5006 & 
sleep 10
# Other commands
export CUDA_VISIBLE_DEVICES=6,7
sh script/dpoplus_script.sh "nouns" "ultra" "http://127.0.0.1:5006/train" 29522 "conf_subsamp_onlynouns"
jobs
pkill -f "conf_subsamp_onlynouns"
jobs
