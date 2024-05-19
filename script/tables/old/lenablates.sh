export BASEMODEL="facebook/opt-125m"
export BETA=0.1
export CUDA_VISIBLE_DEVICES=0
export EXTRAEVAL=""

export BETA=0.1
export CUDA_VISIBLE_DEVICES=4
export LTYPE='norefnolen'
sh script/train_ipo.sh "bagofwords" "bowsynth50knozeros" "" "dponoreflen" 12345

# export LTYPE='sigmoidnolen'
# sh script/train_ipo.sh "bagofwords" "bowsynth50knozeros" "" "nolen" 12346