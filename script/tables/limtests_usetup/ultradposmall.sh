export BASEMODEL="allenai/tulu-2-7b"
export BETA=0.05
export CUDA_VISIBLE_DEVICES=5
# sh script/train_dpo.sh "ultra" "ultra500" "ultratinyheld" "smalldpo" 29526
sh script/train_dpo.sh "ultra" "ultra1k" "ultratinyheld" "smalldpo1k" 29526

