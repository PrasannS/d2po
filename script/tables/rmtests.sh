export CUDA_VISIBLE_DEVICES=0,1
export LR=1e-4
export BASEMODEL="outputs/models/bagofwords/bowtiny_rm"
export NOLORA=False
export REINIT=False
export LTYPE="normal"
export EVFIRST=1
export EXTRAEVAL="outputs/data/bagofwords/"
export BSIZE=2

# sh script/train_rm.sh "ultra" "ultra500" "ultratinyheld" 12350 "tinyrm"
sh script/train_rm.sh "ultra" "ultra1k" "ultratinyheld" 12350 "tinyrm1k"

export EVFIRST=1
