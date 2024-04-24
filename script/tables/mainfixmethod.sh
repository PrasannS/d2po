# SH File for running the whole shebang, fixed method up with more code

export CFG=src/configs/ppo_2gpu.yaml
export STEPS=500
export SUPDATES=10000000
export SEED=0
export KEEPLONG=0
export MLEN=50

defaults() {
    export BASEMODEL="outputs/models/bagofwords/bowtiny_dpo"
    export ATYPE="rand"
    export ULR=1e-4
    export CRATIO=1
}

# BOW examine behavior at smaller intervals
defaults
export SEED=2
export ATYPE="conf"
export UEPOCHS=2
export APBSIZE=16
export GREWARD="bagofwords"

export DPOBATCHSIZE=32
export MBSIZE=32
export GBSIZE=32
export USEDPO=0
export PPOUPDATES=4
export ONLYOLDUPDATES=0

export SAMPN=$((256))
export RELABELS=$((16))
export BASEMODEL="facebook/opt-125m"
export CUDA_VISIBLE_DEVICES=3
# # noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
export SFREQ=25

# export GREWARD="paraphrase"
# nohup sh script/newupdateapi.sh "paraphrase" "" "tiny_rm" "parconf_newalgo_256_16_ndata" 5007 &
# sleep 10
# # Other commands
# export CUDA_VISIBLE_DEVICES=4,5
# sh script/dpoplus_script.sh "paraphrase" "outputs/data/paraphrase/parappoinps" "http://127.0.0.1:5007/train" 29523 "parconf_newalgo_256_16_ndata"
# jobs
# pkill -f "parconf_newalgo_256_16_ndata"
# jobs

# export SAMPN=$((64))
# export RELABELS=$((16))
# export GREWARD="paraphrase"
# export CUDA_VISIBLE_DEVICES=3

# nohup sh script/newupdateapi.sh "paraphrase" "" "tiny_rm" "parconf_newalgo_64_16_ndata" 5007 &
# sleep 10
# # Other commands
# export CUDA_VISIBLE_DEVICES=4,5
# sh script/dpoplus_script.sh "paraphrase" "outputs/data/paraphrase/parappoinps" "http://127.0.0.1:5007/train" 29523 "parconf_newalgo_64_16_ndata"
# jobs
# pkill -f "parconf_newalgo_64_16_ndata"
# jobs
export STEPS=500
export DPOBATCHSIZE=32
export MBSIZE=16
export GBSIZE=16
export BASEMODEL="outputs/models/math/mathbigdata1b"
export MLEN=75
export SAMPN=$((32*64))
export RELABELS=$((32*8))
export GREWARD="math"
export SEED=3
export CUDA_VISIBLE_DEVICES=0
# # noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
# nohup sh script/newupdateapi.sh "math" "" "mathbigdata1b" "1bbig_128_8_seed3" 5005 & 
# # # Other commands
# export CUDA_VISIBLE_DEVICES=1,2
# sh script/dpoplus_script.sh "math" "outputs/data/math/mathppoinps200k" "http://127.0.0.1:5005/train" 29522 "1bbig_128_8_seed3"
# jobs
# pkill -f "1bbig_128_8_seed3"
# jobs

# export CUDA_VISIBLE_DEVICES=0
# export SEED=4
# nohup sh script/newupdateapi.sh "math" "" "mathbigdata1b" "1bbig_128_8_seed4" 5005 & 
# # # Other commands
# export CUDA_VISIBLE_DEVICES=1,2
# sh script/dpoplus_script.sh "math" "outputs/data/math/mathppoinps200k" "http://127.0.0.1:5005/train" 29522 "1bbig_128_8_seed4"
# jobs
# pkill -f "1bbig_128_8_seed4"
# jobs

export STEPS=1000
export DPOBATCHSIZE=8
export MBSIZE=8
export GBSIZE=8
export BASEMODEL="facebook/opt-125m"
export MLEN=50
export SAMPN=$((80))
export RELABELS=$((28))
export GREWARD="unique_nns"

export CUDA_VISIBLE_DEVICES=1
export SEED=4
nohup sh script/newupdateapi.sh "unique_nns" "" "tiny_rm" "noun40_28_seed4" 5005 & 
# # Other commands
export CUDA_VISIBLE_DEVICES=2,4
sh script/dpoplus_script.sh "unique_nns" "ultra" "http://127.0.0.1:5005/train" 29522 "noun40_28_seed4"
jobs
pkill -f "noun40_28_seed4"
jobs

export CUDA_VISIBLE_DEVICES=1
export SEED=5
nohup sh script/newupdateapi.sh "unique_nns" "" "tiny_rm" "noun40_28_seed5" 5005 & 
# # Other commands
export CUDA_VISIBLE_DEVICES=2,4
sh script/dpoplus_script.sh "unique_nns" "ultra" "http://127.0.0.1:5005/train" 29522 "noun40_28_seed5"
jobs
pkill -f "noun40_28_seed5"
jobs

export CUDA_VISIBLE_DEVICES=1
export SEED=6
nohup sh script/newupdateapi.sh "unique_nns" "" "tiny_rm" "noun40_282_seed6" 5005 & 
# # Other commands
export CUDA_VISIBLE_DEVICES=2,4
sh script/dpoplus_script.sh "unique_nns" "ultra" "http://127.0.0.1:5005/train" 29522 "noun40_28_seed6"
jobs
pkill -f "noun40_28_seed6"
jobs


# export BASEMODEL="facebook/opt-125m"
# export MLEN=100
# # TODO note misalignment between SAMPN and RELABELS thing
# export SAMPN=$((32*128))
# export RELABELS=$((32*8))
# export GREWARD="paraphrase"
# export CUDA_VISIBLE_DEVICES=3
# nohup sh script/newupdateapi.sh "paraphrase" "" "tiny_rm" "para128_8" 5007 &
# sleep 10
# # Other commands
# export CUDA_VISIBLE_DEVICES=4,5
# sh script/dpoplus_script.sh "paraphrase" "outputs/data/paraphrase/parappoinps" "http://127.0.0.1:5007/train" 29523 "para128_8"
# jobs
# pkill -f "para128_8"
# jobs


# export BASEMODEL="facebook/opt-125m"
# export MLEN=100
# # TODO note misalignment between SAMPN and RELABELS thing
# export SAMPN=$((32*32))
# export RELABELS=$((32*2))
# export GREWARD="paraphrase"
# export CUDA_VISIBLE_DEVICES=0
# nohup sh script/newupdateapi.sh "paraphrase" "" "tiny_rm" "parconf_newalgo_32_16_justoffpolicy" 5007 &
# sleep 10
# # Other commands
# export CUDA_VISIBLE_DEVICES=1,2
# sh script/dpoplus_script.sh "paraphrase" "outputs/data/paraphrase/parappoinps" "http://127.0.0.1:5007/train" 29523 "parconf_newalgo_32_16_justoffpolicy"
# jobs
# pkill -f "parconf_newalgo_32_16_justoffpolicy"
# jobs

# export SAMPN=$((32))
# export RELABELS=$((32))

# export GREWARD="paraphrase"

# nohup sh script/newupdateapi.sh "paraphrase" "" "mathbigdata1b" "parconf_newalgo_32" 5007 & 
# # Other commands
# export CUDA_VISIBLE_DEVICES=4,5
# sleep 10
# sh script/dpoplus_script.sh "paraphrase" "ultra" "http://127.0.0.1:5007/train" 29523 "parconf_newalgo_32"

# jobs
# pkill -f "parconf_newalgo_32"
# jobs



# export BASEMODEL="outputs/models/math/mathbigdata1b"
# export MLEN=75
# export SAMPN=$((32))
# export RELABELS=$((16))
# export GREWARD="math"
# export CUDA_VISIBLE_DEVICES=0
# # noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
# nohup sh script/newupdateapi.sh "math" "" "mathbigdata1b" "mathconf_newalgo_32_32_fixcode_1b" 5005 & 
# # Other commands
# export CUDA_VISIBLE_DEVICES=1,2
# sh script/dpoplus_script.sh "math" "outputs/data/math/mathppoinps200k" "http://127.0.0.1:5005/train" 29522 "mathconf_newalgo_32_32_fixcode_1b"
# jobs
# pkill -f "mathconf_newalgo_32_32_fixcode_1b"
# jobs
