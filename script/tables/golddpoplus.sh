export CFG=src/configs/ppo_2gpu.yaml
export STEPS=400
export LABTHRESH=0.3
export SUPDATES=10000000
export SEED=0
export KEEPLONG=0
export MLEN=50

defaults() {
    export REDOBATCH=1
    export LABRATIO=0.25
    export ATYPE="rand"
    export ULR=1e-4
    export DPOBATCHSIZE=32
}

# NOTE SMALL CHANCE OF RACE CONDITIONS, WATCH OUT

# CONF SUB_SAMPLING

defaults

export BASEMODEL="outputs/models/ultra/tiny_dpo_tulu"
export MLEN=256
# BATCH SIZE OF 8 instead of 32 (make gold set equivalent basically), how far can we get

export DPOBATCHSIZE=32
export MBSIZE=2
export GBSIZE=8
# sh script/dpoplus_script.sh "ultra" "outputs/data/ultra/smallultrappoinps" "functionbagofwords" 29520 "ultragolddpoplustest2"

# sh script/dpoplus_script.sh "ultra" "outputs/data/ultra/smallultrappoinps" "functionultrafeedbackgold" 29520 "ultragolddpoplusfinal2"

defaults
export APBSIZE=2
export CUDA_VISIBLE_DEVICES=1
export CRATIO=1
export GBSIZE=8
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/noupdateapi.sh "ultra" "" "tiny_onpdata_rm" "tinyonpdata_dpopp" 5008 & 
sleep 10

export CUDA_VISIBLE_DEVICES=6,7

sh script/dpoplus_script.sh "ultra" "outputs/data/ultra/smallultrappoinps" "http://127.0.0.1:5008/predict" 29521 "ultraextrarmppo"

jobs
pkill -f "tinyonpdata_dpopp"
jobs
# sh script/dpoplus_script.sh "ultra" "ultra" "functionultrafeedbackgold" 29520 "ultragolddpoplus"
