export CUDA_VISIBLE_DEVICES=0,1
export LR=1e-4
export NOLORA=False
export REINIT=False
export LTYPE="normal"
export EVFIRST=0
export BSIZE=2
# export EXTRAEVAL="outputs/data/bagofwords/latereval"
# export BASEMODEL="outputs/models/bagofwords/bowtiny_rm"


# # sh script/train_rm.sh "ultra" "ultra500" "ultratinyheld" 12350 "tinyrm"
# sh script/train_rm.sh "bagofwords" "50set1" "smalleval" 12350 "50set1"
# sh script/train_rm.sh "bagofwords" "50set2" "smalleval" 12350 "50set2"

# sh script/train_rm.sh "bagofwords" "50set3" "smalleval" 12350 "50set3"

# sh script/train_rm.sh "bagofwords" "50set4" "smalleval" 12350 "50set4"
# sh script/train_rm.sh "bagofwords" "50set5" "smalleval" 12350 "50set5"

# export EVFIRST=0
# sh script/train_rm.sh "bagofwords" "5set1" "smalleval" 12350 "5set1"
# sh script/train_rm.sh "bagofwords" "5set2" "smalleval" 12350 "5set2"
# sh script/train_rm.sh "bagofwords" "5set3" "smalleval" 12350 "5set3"
# sh script/train_rm.sh "bagofwords" "5set4" "smalleval" 12350 "5set4"
# sh script/train_rm.sh "bagofwords" "5set5" "smalleval" 12350 "5set5"

export EXTRAEVAL="outputs/data/nouns/latereval"
export BASEMODEL="outputs/models/nouns/tiny_rm"

# export BASEMODEL="outputs/models/nouns/smalldpo"
# export BETA=0.05
export CUDA_VISIBLE_DEVICES=0,1
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


export BASEMODEL="outputs/models/contrastivedistill/tiny_rm"
export BETA=0.05
# export CUDA_VISIBLE_DEVICES=5,6
export EXTRAEVAL="outputs/data/contrastivedistill/latereval"


# # # sh script/train_dpo.sh "ultra" "ultra500" "ultratinyheld" "smalldpo" 29526
# # sh script/train_dpo.sh "ultra" "ultra1k" "ultratinyheld" "smalldpo1k" 29526

# export EVFIRST=0
# sh script/train_rm.sh "contrastivedistill" "5set1" "earlyeval" 12351 "5set1" 
sh script/train_rm.sh "contrastivedistill" "5set1" "latereval" 12351 "5set1lat" 
sh script/train_rm.sh "contrastivedistill" "5set2" "latereval" 12351 "5set2lat" 
sh script/train_rm.sh "contrastivedistill" "5set2" "latereval" 12351 "5set2lat" 


sh script/train_rm.sh "contrastivedistill" "50set1" "earlyeval" 12351 "50set1" 
sh script/train_rm.sh "contrastivedistill" "50set2" "earlyeval" 12351 "50set2" 

# sh script/train_rm.sh "contrastivedistill" "50set1" "latereval" 12351 "50set1lat" 

export BASEMODEL="outputs/models/nouns/tiny_rm"
export BETA=0.05
export CUDA_VISIBLE_DEVICES=3,4
export EXTRAEVAL="outputs/data/nouns/latereval"


# # sh script/train_dpo.sh "ultra" "ultra500" "ultratinyheld" "smalldpo" 29526
# sh script/train_dpo.sh "ultra" "ultra1k" "ultratinyheld" "smalldpo1k" 29526

export EVFIRST=0
# sh script/train_rm.sh "nouns" "5set1" "earlyeval" 12351 "5set1" 
# sh script/train_rm.sh "nouns" "5set2" "earlyeval" 12351 "5set2" 
# sh script/train_rm.sh "nouns" "5set3" "earlyeval" 12351 "5set3" 
# sh script/train_rm.sh "nouns" "5set4" "earlyeval" 12351 "5set4" 
# sh script/train_rm.sh "nouns" "5set5" "earlyeval" 12351 "5set5" 


# sh script/train_rm.sh "nouns" "50set1" "earlyeval" 12351 "50set1" 
# sh script/train_rm.sh "nouns" "50set2" "earlyeval" 12351 "50set2" 
# sh script/train_rm.sh "nouns" "50set3" "earlyeval" 12351 "50set3" 
# sh script/train_rm.sh "nouns" "50set4" "earlyeval" 12351 "50set4" 
# sh script/train_rm.sh "nouns" "50set5" "earlyeval" 12351 "50set5" 



# sh script/train_rm.sh "nouns" "5set1" "latereval" 12351 "5set1lat" 

# sh script/train_rm.sh "nouns" "50set1" "earlyeval" 12351 "50set1" 
# sh script/train_rm.sh "nouns" "50set1" "latereval" 12351 "50set1lat" 