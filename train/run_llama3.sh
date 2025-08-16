MODEL=Llama3.1-8B
OUTPUT_DIR=/Llama3.1-8B

MASTER_PORT=29600 \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
swift sft \
    --model ${MODEL} \
    --model_type llama3 \
    --train_type lora \
    --target_modules all-linear \
    --lora_rank 8 \
    --lora_alpha 32 \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 4 \
    --save_strategy epoch \
    --logging_steps 1 \
    --max_length 8192 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --save_total_limit 3 \
    --save_only_model true \
    --output_dir ${OUTPUT_DIR} \
    --deepspeed zero3 \
    --use_liger_kernel true \
    --attn_impl flash_attn \
    --seed 42 \
    --dataset 'thtb.jsonl' \
    --split_dataset_ratio 0.02