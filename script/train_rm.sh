
echo $LR
echo $BASEMODEL
torchrun --nnodes 1  --nproc_per_node 2 --master_port=${4} src/train_rm.py \
    --model_name=$BASEMODEL \
    --output_dir=outputs/checkpoints/${1}/${2}_${5}_rm/ \
    --dataset="outputs/data/${1}/${2}" \
    --rand_ratio=0 \
    --evaldata="outputs/data/${1}/${3}" \
    --balance_len=0 \
    --num_train_epochs=5 \
    --per_device_train_batch_size=$BSIZE \
    --per_device_eval_batch_size=4 \
    --gradient_accumulation_steps=2 \
    --eval_steps=500 \
    --save_steps=500 \
    --learning_rate=$LR \
    --losstype=$LTYPE \
    --nolora=$NOLORA \
    --random_reinit=$REINIT > "outputs/logs/rm/${1}_${3}_rm.out"

