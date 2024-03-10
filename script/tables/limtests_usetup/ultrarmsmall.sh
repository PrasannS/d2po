export CUDA_VISIBLE_DEVICES=6,7
export LR=1e-4
export BASEMODEL="meta-llama/Llama-2-7b-hf"
export NOLORA=False
export REINIT=False
export LTYPE="normal"
export BSIZE=2

# sh script/train_rm.sh "ultra" "ultra500" "ultratinyheld" 12350 "tinyrm"
sh script/train_rm.sh "ultra" "ultra1k" "ultratinyheld" 12350 "tinyrm1k"