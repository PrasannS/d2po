export BASEMODEL="outputs/models/bagofwords/bowtiny_dpo"
export BETA=0.05
export CUDA_VISIBLE_DEVICES=0
export EXTRAEVAL="outputs/data/bagofwords/latereval"

# export EVFIRST=0
# sh script/train_dpo.sh "nouns" "5set1" "earlyeval" "5set1" 12350 
# sh script/train_dpo.sh "nouns" "5set1" "latereval" "5set1lat" 12350 
# # sh script/train_dpo.sh "bagofwords" "5set3" "smalleval" "5set3" 12350 
# # sh script/train_dpo.sh "bagofwords" "5set4" "smalleval" "5set4" 12350 
# # sh script/train_dpo.sh "bagofwords" "5set5" "smalleval" "5set5" 12350 

# # sh script/train_rm.sh "ultra" "ultra500" "ultratinyheld" 12350 "tinyrm"
# sh script/train_dpo.sh "bagofwords" "50set1" "earlyeval" "50set1" 12350
# sh script/train_dpo.sh "bagofwords" "50set2" "latereval" "50set2" 12350


# export BASEMODEL="outputs/models/contrastivedistill/smalldpo"
export BASEMODEL="facebook/opt-125m"
export BETA=0.1
export CUDA_VISIBLE_DEVICES=0
export EXTRAEVAL=""

# # sh script/train_dpo.sh "bagofwords" "bowbase6k" "" "6kdpo" 12341
# export CUDA_VISIBLE_DEVICES=0
# sh script/train_ipo.sh "bagofwords" "bowsynth50knozeros" "" "ipopt1" 12342

# export BETA=100
# export CUDA_VISIBLE_DEVICES=0
# sh script/train_ipo.sh "bagofwords" "bowsynth50knozeros" "" "ipo100" 12342

# export BETA=1
# export CUDA_VISIBLE_DEVICES=1
# sh script/train_ipo.sh "bagofwords" "bowsynth50knozeros" "" "ipo1" 12343

# export BETA=10
# export CUDA_VISIBLE_DEVICES=2
# sh script/train_ipo.sh "bagofwords" "bowsynth50knozeros" "" "ipo10" 12344

export BETA=0.001
export CUDA_VISIBLE_DEVICES=3
sh script/train_ipo.sh "bagofwords" "bowsynth50knozeros" "" "ipopt001" 12345



# # sh script/train_dpo.sh "ultra" "ultra500" "ultratinyheld" "smalldpo" 29526
# sh script/train_dpo.sh "ultra" "ultra1k" "ultratinyheld" "smalldpo1k" 29526

# export EVFIRST=0
# sh script/train_dpo.sh "bagofwords" "bow20k" "" "20kdpo" 12350 


# sh script/train_dpo.sh "contrastivedistill" "5set1" "earlyeval" "5set1" 12350 


# # sh script/train_rm.sh "ultra" "ultra500" "ultratinyheld" 12350 "tinyrm"
# sh script/train_dpo.sh "contrastivedistill" "50set1" "earlyeval" "50set1" 12350
# sh script/train_dpo.sh "contrastivedistill" "50set2" "latereval" "50set2" 12350