import torch
from waynav.data import Low_Level_Dataset
from waynav.data.collate import Low_Level_Collator
from waynav.model import VLN_LL_Action
from waynav.trainer import Low_Level_Trainer
from waynav.eval import compute_confusion_matrix
from transformers import TrainingArguments, AutoConfig
from torch.profiler import profile, record_function, ProfilerActivity

train_data_dir = '/local1/cfyang/low_level_train'
eval_data_dir = '/local1/cfyang/low_level_valid_seen'
unseen_eval_data_dir = '/local1/cfyang/low_level_valid_unseen'
train_dataset = Low_Level_Dataset(train_data_dir)
eval_dataset = Low_Level_Dataset(eval_data_dir)
unseen_eval_dataset = Low_Level_Dataset(unseen_eval_data_dir)
eval_dataset_dict = {
    'seen': eval_dataset,
    'unseen': unseen_eval_dataset
}
tokenizer = train_dataset.tokenizer
data_collator = Low_Level_Collator(tokenizer)

config = AutoConfig.from_pretrained('bert-base-uncased')
# config.update({'num_hidden_layers': 12})
# config.update({'type_vocab_size': 5})

pretrain_weight = 'bert-base-uncased'
# pretrain_weight = "/local1/cfyang/output/subpolicy/ll_action/checkpoint-1000"
model = VLN_LL_Action.from_pretrained(pretrain_weight, config=config)
output_path = "/local1/cfyang/output/subpolicy/ll_new_sub_init"
batch_size = 128
learning_rate = 1e-4
save_steps = 500

training_args = TrainingArguments(
    output_dir=output_path,
    evaluation_strategy="epoch",
    learning_rate=learning_rate,
    warmup_ratio=0.2,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    # dataloader_num_workers=4,
    num_train_epochs=10,
    # max_steps=200,
    weight_decay=0.0001,
    logging_steps=50,
    save_steps=save_steps,
    report_to='tensorboard',
    ddp_find_unused_parameters=False,
    # lr_scheduler_type='constant_with_warmup',
    dataloader_drop_last=True,
)

trainer = Low_Level_Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset_dict,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_confusion_matrix
)

# trainer.evaluate(eval_dataset=unseen_eval_dataset)
trainer.train()

# from pyinstrument import Profiler
# profiler = Profiler()
# profiler.start()
# trainer.evaluate(eval_dataset=unseen_eval_dataset)
# profiler.stop()
# print(profiler.output_text(unicode=True, color=True, show_all=True))