# SH File for running the whole shebang on cdistill, noun tasks, will debug math separately

export CFG=src/configs/ppo_2gpu.yaml
export STEPS=2000
export SUPDATES=10000000
export SEED=2
export KEEPLONG=10

defaults() {
    export BASEMODEL="outputs/models/contrastivedistill/smalldpo"
    export MLEN=50
    export ATYPE="conf"
    export ULR=1e-4
    export CRATIO=1
}

# CDIST GET A BASELINE AT SMALLER INTERV LEVEL
defaults
export ATYPE="rand"
export UEPOCHS=3
export APBSIZE=16
export GREWARD="contrastivedistill"

export DPOBATCHSIZE=32
export MBSIZE=32
export GBSIZE=32

export SAMPN=$((32*10))
export RELABELS=$((5))
export CUDA_VISIBLE_DEVICES=3
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/newupdateapi.sh "contrastivedistill" "" "tiny_rm" "rand_cdist_10_5_activefixseed2" 5010 & 
sleep 20
# Other commands
export CUDA_VISIBLE_DEVICES=4,5
sh script/dpoplus_script.sh "contrastivedistill" "outputs/data/contrastivedistill/wikionpprompts200k" "http://127.0.0.1:5010/train" 29522 "rand_cdist_10_5_activefixseed2"
jobs
pkill -f "rand_cdist_10_5_activefixseed2"
jobs

export SEED=3

# CDIST GET A BASELINE AT SMALLER INTERV LEVEL
defaults
export ATYPE="rand"
export UEPOCHS=3
export APBSIZE=16
export GREWARD="contrastivedistill"

export DPOBATCHSIZE=32
export MBSIZE=32
export GBSIZE=32

export SAMPN=$((32*10))
export RELABELS=$((5))
export CUDA_VISIBLE_DEVICES=3
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/newupdateapi.sh "contrastivedistill" "" "tiny_rm" "rand_cdist_10_5_activefixseed3" 5010 & 
sleep 20
# Other commands
export CUDA_VISIBLE_DEVICES=4,5
sh script/dpoplus_script.sh "contrastivedistill" "outputs/data/contrastivedistill/wikionpprompts200k" "http://127.0.0.1:5010/train" 29522 "rand_cdist_10_5_activefixseed3"
jobs
pkill -f "rand_cdist_10_5_activefixseed3"
jobs

defaults
export ATYPE="rand"
export UEPOCHS=1
export APBSIZE=16
export GREWARD="math"
export BASEMODEL="outputs/models/math/smalldpo"
export MLEN=100

export DPOBATCHSIZE=32
export MBSIZE=32
export GBSIZE=32
export SEED=0
export SAMPN=$((32))
export RELABELS=$((1*2))
export CUDA_VISIBLE_DEVICES=3
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/newupdateapi.sh "math" "" "tiny_rm" "easy4_32_2_try" 5010 &
# Other command
export CUDA_VISIBLE_DEVICES=4,5
sh script/dpoplus_script.sh "math" "outputs/data/math/matheasy4" "http://127.0.0.1:5010/train" 29522 "easy4_32_2_try"
jobs
pkill -f "easy4_32_2_try"
jobs

defaults
export ATYPE="rand"
export UEPOCHS=1
export APBSIZE=16
export GREWARD="math"
export BASEMODEL="outputs/models/math/smalldpo"
export MLEN=100

export DPOBATCHSIZE=32
export MBSIZE=32
export GBSIZE=32
export SEED=2
export SAMPN=$((32))
export RELABELS=$((1*2))
export CUDA_VISIBLE_DEVICES=3
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/newupdateapi.sh "math" "" "tiny_rm" "easy4_32_2_try_seed2" 5010 &
# Other command
export CUDA_VISIBLE_DEVICES=4,5
sh script/dpoplus_script.sh "math" "outputs/data/math/matheasy4" "http://127.0.0.1:5010/train" 29522 "easy4_32_2_try_seed2"
jobs
pkill -f "easy4_32_2_try_seed2"
jobs