# OFF POLICY VERSIONS OF ALL THE ACTIVE THINGS: 
# IMPORTANT BASELINE, WHAT HAPPENS WITH MORE DPO UPDATES ON THE SAME THING
export CFG=src/configs/ppo_2gpu.yaml
export STEPS=2000
export SUPDATES=10000000
export SEED=0
export KEEPLONG=10

# CONTRASTIVE DISTILL
defaults() {
    export BASEMODEL="outputs/models/contrastivedistill/smalldpo"
    export MLEN=50
    export ATYPE="conf"
    export ULR=1e-4
    export CRATIO=1
}

export USEDPO=0
export ONLYOLDUPDATES=0
export PPOUPDATES=8
defaults

# # BEST TECHNIQUE (ALL IN ONE)
# defaults
# export ATYPE="conf"
# export UEPOCHS=3
# export APBSIZE=16
# export GREWARD="contrastivedistill"

# export DPOBATCHSIZE=8
# export MBSIZE=8
# export GBSIZE=8

# export SAMPN=$((32*10))
# export RELABELS=$((5))
# export CUDA_VISIBLE_DEVICES=2
# # noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
# # nohup sh script/newupdateapi.sh "contrastivedistill" "" "tiny_rm" "justoffpolicy_conf_cdist_100_50_activefix" 5010 & 
# # Other commands
# export CUDA_VISIBLE_DEVICES=0,1
# sh script/dpoplus_script.sh "contrastivedistill" "outputs/data/contrastivedistill/wikionpprompts200k" "functioncontrastivedistill" 29523 "contdistb8_moreupdates_4ups"
# jobs
# # pkill -f "justoffpolicy_conf_cdist_100_50_activefix"
# jobs

# NOUNS ___________
defaults
# export BASEMODEL="outputs/models/nouns/smalldpo"

export ATYPE="conf"
export UEPOCHS=2
export APBSIZE=8
export GREWARD="nouns"

export DPOBATCHSIZE=8
export MBSIZE=8
export GBSIZE=8

# export SAMPN=$((32*5))
# export RELABELS=$((2))
# export CUDA_VISIBLE_DEVICES=2
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
# nohup sh script/newupdateapi.sh "nouns" "" "tiny_rm" "justoffpolicy_confnoun_newalgo_100small_200testv3" 5010 & 
# sleep 10
# Other commands
export CUDA_VISIBLE_DEVICES=2,3
# sh script/dpoplus_script.sh "nouns" "ultra" "functionnouns" 29524 "confnoun_goldb8_4ups"
jobs
# pkill -f "justoffpolicy_confnoun_newalgo_100small_200testv3"
jobs

# #  __________________--
defaults
export BASEMODEL="outputs/models/bagofwords/bowtiny_dpo"
# export ATYPE="conf"
# export UEPOCHS=3
# export APBSIZE=16
# export GREWARD="bagofwords"

export DPOBATCHSIZE=8
export MBSIZE=8
export GBSIZE=8

# export SAMPN=$((32*5))
# export RELABELS=$((1*5))
# export CUDA_VISIBLE_DEVICES=2
# # # noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
# # nohup sh script/newupdateapi.sh "bagofwords" "" "bowtiny_rm" "justoffpolicy_conf_newalgo_500" 5010 & 
# # # Other commands
# export CUDA_VISIBLE_DEVICES=4,5
sh script/dpoplus_script.sh "bagofwords" "ultra" "functionbagofwords" 29526 "conf_newalgo_goldb8_8ups"
# jobs
# # pkill -f "justoffpolicy_conf_newalgo_500"
# jobs

# defaults
# export PPOUPDATES=4
# export BASEMODEL="outputs/models/bagofwords/bowtiny_dpo"
# export ATYPE="conf"
# export UEPOCHS=3
# export APBSIZE=16
# export GREWARD="bagofwords"

# export DPOBATCHSIZE=8
# export MBSIZE=8
# export GBSIZE=8

# export SAMPN=$((32*5))
# export RELABELS=$((1*5))
# export CUDA_VISIBLE_DEVICES=2
# # # noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
# # nohup sh script/newupdateapi.sh "bagofwords" "" "bowtiny_rm" "justoffpolicy_conf_newalgo_500" 5010 & 
# # # Other commands
# export CUDA_VISIBLE_DEVICES=4,5
# sh script/dpoplus_script.sh "bagofwords" "ultra" "functionbagofwords" 29527 "conf_newalgo_goldb8_1ups"
# jobs
# # pkill -f "justoffpolicy_conf_newalgo_500"
# jobs

# ___________________-

# BEST TECHNIQUE (ALL IN ONE)
# defaults
# export ATYPE="conf"
# export UEPOCHS=3
# export APBSIZE=16
# export GREWARD="contrastivedistill"

# export DPOBATCHSIZE=32
# export MBSIZE=32
# export GBSIZE=32

# export SAMPN=$((32*10))
# export RELABELS=$((5))
# export CUDA_VISIBLE_DEVICES=0
# # noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
# nohup sh script/newupdateapi.sh "contrastivedistill" "" "tiny_rm" "justoffpolicy_conf_cdist_10_50_activefix" 5001 & 
# sleep 20
# # Other commands
# export CUDA_VISIBLE_DEVICES=1,2
# sh script/dpoplus_script.sh "contrastivedistill" "outputs/data/contrastivedistill/wikionpprompts200k" "http://127.0.0.1:5001/train" 29523 "justoffpolicy_conf_cdist_10_50_activefix"
# jobs
# pkill -f "justoffpolicy_conf_cdist_10_50_activefix"
# jobs

# # NOUNS
# defaults
# export BASEMODEL="outputs/models/nouns/smalldpo"

# export ATYPE="conf"
# export UEPOCHS=3
# export APBSIZE=16
# export GREWARD="nouns"

# export DPOBATCHSIZE=32
# export MBSIZE=32
# export GBSIZE=32

# export SAMPN=$((32*5))
# export RELABELS=$((2))
# export CUDA_VISIBLE_DEVICES=3
# # noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
# nohup sh script/newupdateapi.sh "nouns" "" "tiny_rm" "justoffpolicy_confnoun_newalgo_5_2" 5002 & 
# sleep 20
# # Other commands
# export CUDA_VISIBLE_DEVICES=6,7
# sh script/dpoplus_script.sh "nouns" "ultra" "http://127.0.0.1:5002/train" 29524 "justoffpolicy_confnoun_newalgo_5_2"
# jobs
# pkill -f "justoffpolicy_confnoun_newalgo_5_2"
# jobs

# defaults

# export BASEMODEL="outputs/models/bagofwords/bowtiny_dpo"
# export ATYPE="conf"
# export UEPOCHS=3
# export APBSIZE=16
# export GREWARD="bagofwords"

# export DPOBATCHSIZE=32
# export MBSIZE=32
# export GBSIZE=32

# export SAMPN=$((32*5))
# export RELABELS=$((1*5))
# export CUDA_VISIBLE_DEVICES=3
# # noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
# nohup sh script/newupdateapi.sh "bagofwords" "" "bowtiny_rm" "justoffpolicy_conf_newalgo_5_5" 5003 & 
# # Other commands
# export CUDA_VISIBLE_DEVICES=4,5
# sh script/dpoplus_script.sh "bagofwords" "ultra" "http://127.0.0.1:5003/train" 29525 "justoffpolicy_conf_newalgo_5_5"
# jobs
# pkill -f "justoffpolicy_conf_newalgo_5_5"
# jobs
