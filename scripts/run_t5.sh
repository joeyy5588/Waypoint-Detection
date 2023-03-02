python run_t5.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --train_file /local1/cfyang/alfworld/data/json_2.1.1/train/interaction_dataset.json \
    --validation_file /local1/cfyang/alfworld/data/json_2.1.1/valid_seen/interaction_dataset.json \
    --source_prefix "" \
    --output_dir /local1/cfyang/output/subpolicy/interaction \
    --overwrite_output_dir \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate
    