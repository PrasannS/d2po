export BASEMODEL="outputs/models/math/mathbigdata1b"

export BETA=0.1
export CUDA_VISIBLE_DEVICES=1,2,3,4
# export EXTRAEVAL="outputs/data/bagofwords/latereval"
# export BASEMODEL="allenai/tulu-2-7b"


# export EVFIRST=0
# sh script/train_dpo.sh "easymusr" "easymusrpref" "easymusrpref" "easymusrdpocheck" 12350 
sh script/train_dpo.sh "math" "fgpref200k" "fgheld" "finegrainmathdpo" 12350 


# sh script/train_dpo.sh "nouns" "5set1" "latereval" "5set1lat" 12350 
# # sh script/train_dpo.sh "bagofwords" "5set3" "smalleval" "5set3" 12350 
# # sh script/train_dpo.sh "bagofwords" "5set4" "smalleval" "5set4" 12350 
# # sh script/train_dpo.sh "bagofwords" "5set5" "smalleval" "5set5" 12350 

# # sh script/train_rm.sh "ultra" "ultra500" "ultratinyheld" 12350 "tinyrm"
# sh script/train_dpo.sh "bagofwords" "50set1" "earlyeval" "50set1" 12350
# sh script/train_dpo.sh "bagofwords" "50set2" "latereval" "50set2" 12350


# export BASEMODEL="outputs/models/contrastivedistill/smalldpo"
export BASEMODEL="/u/prasanns/research/active-rlhf/outputs/models/ultra/tiny_dpo_tulu"
export BETA=0.1
export CUDA_VISIBLE_DEVICES=0
export EXTRAEVAL=""
