export BASEMODEL="allenai/tulu-2-7b"
export BETA=0.05
export CUDA_VISIBLE_DEVICES=5
sh script/train_dpo.sh "ultra" "ultra5k" "ultratinyheld" "smalldpo" 29526
