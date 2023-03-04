CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/finetune_t5.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --train_file /local1/cfyang/alfworld/data/json_2.1.1/train/instruction_type.json \
    --validation_file /local1/cfyang/alfworld/data/json_2.1.1/valid_unseen/instruction_type.json \
    --source_prefix "" \
    --output_dir /local1/cfyang/output/subpolicy/instruction_type \
    --overwrite_output_dir \
    --per_device_train_batch_size=64 \
    --per_device_eval_batch_size=64 \
    --predict_with_generate
    