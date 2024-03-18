export CFG=src/configs/ppo_2gpu.yaml
export STEPS=235
export SUPDATES=10000000
export SEED=0
export KEEPLONG=0

defaults() {
    export BASEMODEL="outputs/models/ultra/tiny_dpo_tulu"
    export MLEN=256
    export ATYPE="rand"
    export ULR=1e-4
    export CRATIO=1
}

# defaults
# export ATYPE="conf"
# export UEPOCHS=2
# export APBSIZE=1
# export GREWARD="ultrafeedbackgold"

# export DPOBATCHSIZE=32
# export MBSIZE=2
# export GBSIZE=8

# # every 40 steps of training (total 235), get 100 preferences out of a total possible 1280 preferences
# export SAMPN=$((32*2))
# export RELABELS=$((2))
# export CUDA_VISIBLE_DEVICES=5
# # noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
# nohup sh script/newupdateapi.sh "ultra" "" "tiny_rm" "conf_active_newalgoultratest" 5009 &
# # Other command
# export CUDA_VISIBLE_DEVICES=6,7
# sh script/dpoplus_script.sh "ultra" "outputs/data/ultra/smallultrappoinps" "http://127.0.0.1:5009/train" 29523 "conf_active_newalgoultratest"
# jobs
# pkill -f "conf_active_newalgoultratest"
# jobs

defaults
export ATYPE="conf"
export UEPOCHS=3
export APBSIZE=1
export GREWARD="ultrafeedbackgold"

export DPOBATCHSIZE=32
export MBSIZE=2
export GBSIZE=8

# every 40 steps of training (total 235), get 100 preferences out of a total possible 1280 preferences
export SAMPN=$((32*80))
export RELABELS=$((100))
export CUDA_VISIBLE_DEVICES=5
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/newupdateapi.sh "ultra" "" "tiny_rm" "conf_active_newalgoultra" 5009 &
# Other command
export CUDA_VISIBLE_DEVICES=6,7
sh script/dpoplus_script.sh "ultra" "outputs/data/ultra/smallultrappoinps" "http://127.0.0.1:5009/train" 29523 "conf_active_newalgoultra"
jobs
pkill -f "conf_active_newalgoultra"
jobs