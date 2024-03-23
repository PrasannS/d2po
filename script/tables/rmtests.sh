export CUDA_VISIBLE_DEVICES=0,1
export LR=1e-4
export BASEMODEL="outputs/models/bagofwords/bowtiny_rm"
export NOLORA=False
export REINIT=False
export LTYPE="normal"
export EVFIRST=0
export EXTRAEVAL="outputs/data/bagofwords/latereval"
export BSIZE=2

# sh script/train_rm.sh "ultra" "ultra500" "ultratinyheld" 12350 "tinyrm"
sh script/train_rm.sh "bagofwords" "50set1" "smalleval" 12350 "50set1"
sh script/train_rm.sh "bagofwords" "50set2" "smalleval" 12350 "50set2"

sh script/train_rm.sh "bagofwords" "50set3" "smalleval" 12350 "50set3"

sh script/train_rm.sh "bagofwords" "50set4" "smalleval" 12350 "50set4"
sh script/train_rm.sh "bagofwords" "50set5" "smalleval" 12350 "50set5"

export EVFIRST=0
sh script/train_rm.sh "bagofwords" "5set1" "smalleval" 12350 "5set1"
sh script/train_rm.sh "bagofwords" "5set2" "smalleval" 12350 "5set2"
sh script/train_rm.sh "bagofwords" "5set3" "smalleval" 12350 "5set3"
sh script/train_rm.sh "bagofwords" "5set4" "smalleval" 12350 "5set4"
sh script/train_rm.sh "bagofwords" "5set5" "smalleval" 12350 "5set5"
