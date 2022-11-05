import torch
from waynav.data import Action_Dataset
from waynav.data.collate import Action_Collator
from waynav.model import VLN_Navigator
from waynav.trainer import ActionTrainer
from transformers import TrainingArguments, AutoConfig
from torch.profiler import profile, record_function, ProfilerActivity

train_data_dir = '/data/joey/substep_train'
eval_data_dir = '/data/joey/substep_valid_seen'
train_dataset = Action_Dataset(train_data_dir)
eval_dataset = Action_Dataset(eval_data_dir)
tokenizer = train_dataset.tokenizer
data_collator = Action_Collator(tokenizer)

config = AutoConfig.from_pretrained('bert-base-uncased')
config.update({'num_hidden_layers': 4})
config.update({'type_vocab_size': 5})

model = VLN_Navigator(config)
output_path = "/data/joey/waypoint/output/action_no_pretrain"
batch_size = 64
learning_rate = 1e-4
save_steps = 500


training_args = TrainingArguments(
    output_dir=output_path,
    evaluation_strategy="epoch",
    learning_rate=learning_rate,
    warmup_ratio=0.1,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    # dataloader_num_workers=4,
    num_train_epochs=20,
    # max_steps=200,
    weight_decay=0.0001,
    logging_steps=20,
    save_steps=save_steps,
    report_to='tensorboard'
)

trainer = ActionTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)


# trainer.train()

from pyinstrument import Profiler
profiler = Profiler()
profiler.start()
trainer.train()
profiler.stop()
print(profiler.output_text(unicode=True, color=True, show_all=True))

# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
#     train_dataloader = trainer.get_train_dataloader()
#     for (i, inputs) in enumerate(train_dataloader):
#         print(i)


# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
# print(prof.key_averages(group_by_stack_n=5).table(sort_by="cpu_time_total", row_limit=20))



