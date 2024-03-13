export CUDA_VISIBLE_DEVICES=6,7
export LR=1e-4
export BASEMODEL="/u/prasanns/research/active-rlhf/outputs/models/ultra/tiny_rm"
export NOLORA=False
export REINIT=False
export LTYPE="normal"
export BSIZE=2

# sh script/train_rm.sh "ultra" "ultra500" "ultratinyheld" 12350 "tinyrm"
sh script/train_rm.sh "ultra" "dpopgoldannots" "ood200test" 12350 "onprmsmallprefs"