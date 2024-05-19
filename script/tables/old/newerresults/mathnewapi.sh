# SH File for running the whole shebang, multiple jobs

export CFG=src/configs/ppo_2gpu.yaml
export STEPS=2000
export SUPDATES=10000000
export SEED=0
export KEEPLONG=0

defaults() {
    export BASEMODEL="outputs/models/math/smalldpo"
    export MLEN=100
    export ATYPE="rand"
    export ULR=1e-4
    export CRATIO=1
}

# BEST TECHNIQUE (ALL IN ONE)
defaults
export ATYPE="conf"
export UEPOCHS=3
export APBSIZE=16
export GREWARD="math"

export DPOBATCHSIZE=32
export MBSIZE=32
export GBSIZE=32

export SAMPN=$((32*1))
export RELABELS=$((8))
export CUDA_VISIBLE_DEVICES=5
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/newupdateapi.sh "math" "" "tiny_rm" "conf8rep" 5001 & 
# Other command
export CUDA_VISIBLE_DEVICES=6,7
sh script/dpoplus_script.sh "math" "outputs/data/math/mathppoinps200k" "http://127.0.0.1:5001/train" 29510 "conf8rep"
jobs
pkill -f "conf8rep"
jobs

# BEST TECHNIQUE (ALL IN ONE)
defaults
export ATYPE="conf"
export UEPOCHS=3
export APBSIZE=16
export GREWARD="math"

export DPOBATCHSIZE=32
export MBSIZE=32
export GBSIZE=32

export SAMPN=$((32*1))
export RELABELS=$((16))
export CUDA_VISIBLE_DEVICES=5
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/newupdateapi.sh "math" "" "tiny_rm" "conf16rep" 5001 & 
# Other command
export CUDA_VISIBLE_DEVICES=6,7
sh script/dpoplus_script.sh "math" "outputs/data/math/mathppoinps200k" "http://127.0.0.1:5001/train" 29510 "conf16rep"
jobs
pkill -f "conf16rep"
jobs

# BEST TECHNIQUE (ALL IN ONE)
defaults
export ATYPE="conf"
export UEPOCHS=3
export APBSIZE=16
export GREWARD="math"

export DPOBATCHSIZE=32
export MBSIZE=32
export GBSIZE=32

export SAMPN=$((32*1))
export RELABELS=$((2))
export SEED=1
export CUDA_VISIBLE_DEVICES=5
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/newupdateapi.sh "math" "" "tiny_rm" "conf2repseed1" 5001 & 
# Other command
export CUDA_VISIBLE_DEVICES=6,7
sh script/dpoplus_script.sh "math" "outputs/data/math/mathppoinps200k" "http://127.0.0.1:5001/train" 29510 "conf2repseed1"
jobs
pkill -f "conf2repseed1"
jobs

# BEST TECHNIQUE (ALL IN ONE)
defaults
export ATYPE="conf"
export UEPOCHS=3
export APBSIZE=16
export GREWARD="math"

export DPOBATCHSIZE=32
export MBSIZE=32
export GBSIZE=32

export SAMPN=$((32*1))
export RELABELS=$((2))
export SEED=3
export CUDA_VISIBLE_DEVICES=5
# noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
nohup sh script/newupdateapi.sh "math" "" "tiny_rm" "conf2repseed3" 5001 & 
# Other command
export CUDA_VISIBLE_DEVICES=6,7
sh script/dpoplus_script.sh "math" "outputs/data/math/mathppoinps200k" "http://127.0.0.1:5001/train" 29510 "conf2repseed3"
jobs
pkill -f "conf2repconf2repseed3seed1"
jobs

# defaults
# export ATYPE="conf"
# export UEPOCHS=3
# export APBSIZE=16
# export GREWARD="math"

# export DPOBATCHSIZE=32
# export MBSIZE=32
# export GBSIZE=32

# export SAMPN=$((32*100))
# export RELABELS=$((4*100))
# export CUDA_VISIBLE_DEVICES=7
# # noupdateapi "bagofwords" "bowsynth50knozeros" "bowtiny_rm" "reprodtest" 5000
# nohup sh script/newupdateapi.sh "math" "" "tiny_rm" "conf_newalgo_100_4test" 5001 & 
# # Other commands
# export CUDA_VISIBLE_DEVICES=2,3
# sh script/dpoplus_script.sh "math" "outputs/data/math/mathppoinps" "http://127.0.0.1:5001/train" 29510 "conf_newalgo_100_4test"
# jobs
# pkill -f "conf_newalgo_100_4test"
# jobs