# Finetune a custom dataset on T5-small
import torch
from transformers import Seq2SeqTrainingArguments, AutoConfig, Seq2SeqTrainer, T5ForConditionalGeneration, T5Tokenizer
from transformers import DataCollatorForSeq2Seq
from datasets import load_dataset
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np

train_data_dir = '/local1/cfyang/alfworld/data/json_2.1.1/train/interaction_dataset.json'
eval_data_dir = '/local1/cfyang/alfworld/data/json_2.1.1/valid_seen/interaction_dataset.json'
unseen_eval_data_dir = '/local1/cfyang/alfworld/data/json_2.1.1/valid_unseen/interaction_dataset.json'
train_dataset = load_dataset(train_data_dir)
eval_dataset = load_dataset(eval_data_dir)
unseen_eval_dataset = load_dataset(unseen_eval_data_dir)
eval_dataset_dict = {
    'seen': eval_dataset,
    'unseen': unseen_eval_dataset
}
for i in range(10):
    print(train_dataset.__getitem__(i))
tokenizer = train_dataset.tokenizer
data_collator = Subpolicy_Collator(tokenizer)

config = AutoConfig.from_pretrained('t5-small')

pretrain_weight = 'facebook/bart-base'
output_path = "/local1/cfyang/output/subpolicy/new_subpolicy_3x_inst"
# output_path = "/local1/cfyang/output/subpolicy/inference"
batch_size = 128
learning_rate = 1e-4
save_steps = 500

training_args = Seq2SeqTrainingArguments(
    output_dir=output_path,
    evaluation_strategy="epoch",
    learning_rate=learning_rate,
    warmup_ratio=0.2,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    # dataloader_num_workers=4,
    num_train_epochs=40,
    # max_steps=200,
    weight_decay=0.0001,
    logging_steps=20,
    save_steps=save_steps,
    report_to='tensorboard',
    ddp_find_unused_parameters=False,
    # lr_scheduler_type='constant_with_warmup',
    dataloader_drop_last=True,
    predict_with_generate=True,
    generation_max_length=36,
    generation_num_beams=5,
)

trainer = SubpolicyTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset_dict,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
