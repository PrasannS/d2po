# SH File for running the whole shebang, multiple jobs

export CFG=src/configs/ppo_2gpu.yaml
export STEPS=5
export LABTHRESH=0.3
export SUPDATES=10000000
export SEED=0
export KEEPLONG=0
export MLEN=50
export BASEMODEL="facebook/opt-125m"

# RANDOM SUBSAMPLING RUN 

export CUDA_VISIBLE_DEVICES=2
# noupdateapi "bagofwords" "bowsynth50knozeros" "expbow50" "reprodtest" 5000
nohup sh script/updateapi.sh "bagofwords" "bowsynth50knozeros" "expbow50" "rand_subsamp" 5001 & 
# Other commands
export CUDA_VISIBLE_DEVICES=3,4
sh script/dpoplus_script.sh "bagofwords" "ultra" "http://127.0.0.1:5001/train" 29516 "rand_subsamp"
jobs
pkill -f "rand_subsamp"
jobs

# RANDOM SUBSAMPLING RUN from base (not using DPO)

export CUDA_VISIBLE_DEVICES=2
# noupdateapi "bagofwords" "bowsynth50knozeros" "expbow50" "reprodtest" 5000
nohup sh script/updateapi.sh "bagofwords" "bowsynth50knozeros" "expbow50" "rand_subsamp_obase" 5001 & 
# Other commands
export CUDA_VISIBLE_DEVICES=3,4
sh script/dpoplus_script.sh "bagofwords" "ultra" "http://127.0.0.1:5001/train" 29516 "rand_subsamp_obase"
jobs
pkill -f "rand_subsamp_obase"
jobs

# RAND SUBSAMPLE WITH LOWER LR

export CUDA_VISIBLE_DEVICES=2
# noupdateapi "bagofwords" "bowsynth50knozeros" "expbow50" "reprodtest" 5000
nohup sh script/updateapi.sh "bagofwords" "bowsynth50knozeros" "expbow50" "rand_subsamp_lowlr" 5001 & 
# Other commands
export CUDA_VISIBLE_DEVICES=3,4
sh script/dpoplus_script.sh "bagofwords" "ultra" "http://127.0.0.1:5001/train" 29516 "rand_subsamp_lowlr"
jobs
pkill -f "rand_subsamp_lowlr"
jobs

# CONF SUB_SAMPLING

export CUDA_VISIBLE_DEVICES=2
# noupdateapi "bagofwords" "bowsynth50knozeros" "expbow50" "reprodtest" 5000
nohup sh script/updateapi.sh "bagofwords" "bowsynth50knozeros" "expbow50" "conf_subsamp" 5001 & 
# Other commands
export CUDA_VISIBLE_DEVICES=3,4
sh script/dpoplus_script.sh "bagofwords" "ultra" "http://127.0.0.1:5001/train" 29516 "conf_subsamp"
jobs
pkill -f "conf_subsamp"
jobs

# RAND SUB_SAMP + REPLAY (5 batch)

export CUDA_VISIBLE_DEVICES=5
# noupdateapi "bagofwords" "bowsynth50knozeros" "expbow50" "reprodtest" 5000
nohup sh script/updateapi.sh "bagofwords" "bowsynth50knozeros" "expbow50" "rand_subsamp_replay5" 5001 & 
# Other commands
export CUDA_VISIBLE_DEVICES=6,7
sh script/dpoplus_script.sh "bagofwords" "ultra" "http://127.0.0.1:5001/train" 29516 "rand_subsamp_replay5"
jobs
pkill -f "rand_subsamp"
jobs


# RAND SUB_SAMP + REPLAY (15 batch)

export CUDA_VISIBLE_DEVICES=5
# noupdateapi "bagofwords" "bowsynth50knozeros" "expbow50" "reprodtest" 5000
nohup sh script/updateapi.sh "bagofwords" "bowsynth50knozeros" "expbow50" "rand_subsamp_replay15" 5001 & 
# Other commands
export CUDA_VISIBLE_DEVICES=6,7
sh script/dpoplus_script.sh "bagofwords" "ultra" "http://127.0.0.1:5001/train" 29516 "rand_subsamp_replay15"
jobs
pkill -f "rand_subsamp"
jobs

# RAND SUB_SAMP + IN-DOMAIN PROMPT DATA (maybe with expbow)

export CUDA_VISIBLE_DEVICES=5
# noupdateapi "bagofwords" "bowsynth50knozeros" "expbow50" "reprodtest" 5000
nohup sh script/updateapi.sh "bagofwords" "bowsynth50knozeros" "expbow50" "rand_subsamp_indomain" 5001 & 
# Other commands
export CUDA_VISIBLE_DEVICES=6,7
sh script/dpoplus_script.sh "bagofwords" "ultra" "http://127.0.0.1:5001/train" 29516 "rand_subsamp_indomain"
jobs
pkill -f "rand_subsamp"
jobs

# BEST TECHNIQUE (ALL IN ONE)

export CUDA_VISIBLE_DEVICES=5
# noupdateapi "bagofwords" "bowsynth50knozeros" "expbow50" "reprodtest" 5000
nohup sh script/updateapi.sh "bagofwords" "bowsynth50knozeros" "expbow50" "conf_all" 5001 & 
# Other commands
export CUDA_VISIBLE_DEVICES=6,7
sh script/dpoplus_script.sh "bagofwords" "ultra" "http://127.0.0.1:5001/train" 29516 "conf_all"
jobs
pkill -f "conf_all"
jobs