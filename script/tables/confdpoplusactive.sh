export CFG=src/configs/ppo_2gpu.yaml
export STEPS=125
export LABTHRESH=0.3
export SUPDATES=10000000
export SEED=0
export KEEPLONG=0

defaults() {
    export REDOBATCH=1
    export LABRATIO=0.125
    export ATYPE="conf"
    export ULR=1e-4
    export DPOBATCHSIZE=32
}

# NOTE SMALL CHANCE OF RACE CONDITIONS, WATCH OUT

# CONF SUB_SAMPLING

defaults
export APBSIZE=16
export CUDA_VISIBLE_DEVICES=3
export GREWARD="bagofwordsultra"
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/updateapi.sh "ultra" "" "tiny_rm" "conf_active_ultra" 5006 & 
sleep 10

export BASEMODEL="outputs/models/ultra/tiny_dpo_tulu"
export MLEN=256
# BATCH SIZE OF 8 instead of 32 (make gold set equivalent basically), how far can we get
export CUDA_VISIBLE_DEVICES=1,2

export DPOBATCHSIZE=32
export MBSIZE=2
# sh script/dpoplus_script.sh "ultra" "outputs/data/ultra/smallultrappoinps" "functionbagofwords" 29520 "ultragolddpoplustest2"

sh script/dpoplus_script.sh "ultra" "outputs/data/ultra/smallultrappoinps" "http://127.0.0.1:5006/train" 29520 "confactivegoldlabels"

# sh script/dpoplus_script.sh "ultra" "ultra" "functionultrafeedbackgold" 29520 "ultragolddpoplus"
