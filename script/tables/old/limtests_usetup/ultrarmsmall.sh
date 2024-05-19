export CUDA_VISIBLE_DEVICES=1,2
export LR=1e-4
export BASEMODEL="facebook/opt-125m"
export NOLORA=False
export REINIT=False
export LTYPE="normal"
export BSIZE=32
export EVFIRST=1

# # sh script/train_rm.sh "ultra" "ultra500" "ultratinyheld" 12350 "tinyrm"
sh script/train_rm.sh "unique_nns" "fullnpref" "uniqueval" 12351 "bigrmnew40k"

# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12350 src/train_rm.py \
#     --model_name=$BASEMODEL \
#     --output_dir=outputs/checkpoints/${1}/${2}_${5}_rm/ \
#     --dataset="outputs/data/${1}/${2}" \
#     --rand_ratio=0 \
#     --evaldata="outputs/data/${1}/${3}" \
#     --balance_len=0 \
#     --num_train_epochs=3 \
#     --per_device_train_batch_size=$BSIZE \
#     --per_device_eval_batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --eval_steps=20 \
#     --save_steps=20 \
#     --learning_rate=$LR \
#     --eval_first_step=$EVFIRST \
#     --extraevaldata=$EXTRAEVAL \
#     --losstype=$LTYPE \
#     --nolora=$NOLORA \
#     --random_reinit=$REINIT > "outputs/logs/rm/${1}_${2}_${3}_rm.out"