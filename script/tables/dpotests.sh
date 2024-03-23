export BASEMODEL="outputs/models/bagofwords/bowtiny_dpo"
export BETA=0.05
export CUDA_VISIBLE_DEVICES=5
export EXTRAEVAL="outputs/data/bagofwords/latereval"


# # sh script/train_dpo.sh "ultra" "ultra500" "ultratinyheld" "smalldpo" 29526
# sh script/train_dpo.sh "ultra" "ultra1k" "ultratinyheld" "smalldpo1k" 29526

export EVFIRST=0
# sh script/train_dpo.sh "bagofwords" "5set1" "latereval" "5set1" 12350 
# sh script/train_dpo.sh "bagofwords" "5set2" "latereval" "5set2" 12350 
# # sh script/train_dpo.sh "bagofwords" "5set3" "smalleval" "5set3" 12350 
# # sh script/train_dpo.sh "bagofwords" "5set4" "smalleval" "5set4" 12350 
# # sh script/train_dpo.sh "bagofwords" "5set5" "smalleval" "5set5" 12350 

# # sh script/train_rm.sh "ultra" "ultra500" "ultratinyheld" 12350 "tinyrm"
# sh script/train_dpo.sh "bagofwords" "50set1" "latereval" "50set1" 12350
# sh script/train_dpo.sh "bagofwords" "50set2" "latereval" "50set2" 12350

# sh script/train_dpo.sh "bagofwords" "50set3" "smalleval" "50set3" 12350

# sh script/train_dpo.sh "bagofwords" "50set4" "smalleval" "50set4" 12350
# sh script/train_dpo.sh "bagofwords" "50set5" "smalleval" "50set5" 12350


# export BASEMODEL="outputs/models/nouns/smalldpo"
# export BETA=0.05
# export CUDA_VISIBLE_DEVICES=5
# export EXTRAEVAL="outputs/data/nouns/latereval"


# # # sh script/train_dpo.sh "ultra" "ultra500" "ultratinyheld" "smalldpo" 29526
# # sh script/train_dpo.sh "ultra" "ultra1k" "ultratinyheld" "smalldpo1k" 29526

# export EVFIRST=0
# sh script/train_dpo.sh "nouns" "5set1" "earlyeval" "5set1" 12350 
# sh script/train_dpo.sh "nouns" "5set1" "latereval" "5set1lat" 12350 
# # sh script/train_dpo.sh "bagofwords" "5set3" "smalleval" "5set3" 12350 
# # sh script/train_dpo.sh "bagofwords" "5set4" "smalleval" "5set4" 12350 
# # sh script/train_dpo.sh "bagofwords" "5set5" "smalleval" "5set5" 12350 

# # sh script/train_rm.sh "ultra" "ultra500" "ultratinyheld" 12350 "tinyrm"
# sh script/train_dpo.sh "bagofwords" "50set1" "earlyeval" "50set1" 12350
# sh script/train_dpo.sh "bagofwords" "50set2" "latereval" "50set2" 12350


export BASEMODEL="outputs/models/contrastivedistill/smalldpo"
export BETA=0.05
export CUDA_VISIBLE_DEVICES=5
export EXTRAEVAL="outputs/data/contrastivedistill/latereval"


# # sh script/train_dpo.sh "ultra" "ultra500" "ultratinyheld" "smalldpo" 29526
# sh script/train_dpo.sh "ultra" "ultra1k" "ultratinyheld" "smalldpo1k" 29526

export EVFIRST=0
sh script/train_dpo.sh "contrastivedistill" "5set1" "earlyeval" "5set1" 12350 
sh script/train_dpo.sh "contrastivedistill" "5set1" "latereval" "5set1lat" 12350 
# sh script/train_dpo.sh "bagofwords" "5set3" "smalleval" "5set3" 12350 
# sh script/train_dpo.sh "bagofwords" "5set4" "smalleval" "5set4" 12350 
# sh script/train_dpo.sh "bagofwords" "5set5" "smalleval" "5set5" 12350 

# sh script/train_rm.sh "ultra" "ultra500" "ultratinyheld" 12350 "tinyrm"
sh script/train_dpo.sh "contrastivedistill" "50set1" "earlyeval" "50set1" 12350
sh script/train_dpo.sh "contrastivedistill" "50set2" "latereval" "50set2" 12350