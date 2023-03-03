import torch
from waynav.data import Subpolicy_Dataset, Subpolicy_NoImg_Dataset
from waynav.data.collate import Subpolicy_Collator, Subpolicy_NoImg_Collator
from waynav.model import BartForSubpolicyGeneration, BartNoImgForSubpolicyGeneration
from waynav.trainer import SubpolicyTrainer
from waynav.eval import compute_meta_action_metrics
from transformers import Seq2SeqTrainingArguments, AutoConfig
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
from tqdm import tqdm

train_data_dir = '/local1/cfyang/substep_train'
eval_data_dir = '/local1/cfyang/substep_valid_seen'
unseen_eval_data_dir = '/local1/cfyang/substep_valid_unseen'
train_dataset = Subpolicy_Dataset(train_data_dir)
eval_dataset = Subpolicy_Dataset(eval_data_dir)
# for i in tqdm(range(len(train_dataset))):
#     train_dataset.__getitem__(i)
unseen_eval_dataset = Subpolicy_Dataset(unseen_eval_data_dir)
eval_dataset_dict = {
    'seen': eval_dataset,
    'unseen': unseen_eval_dataset
}
tokenizer = train_dataset.tokenizer
data_collator = Subpolicy_Collator(tokenizer)

config = AutoConfig.from_pretrained('facebook/bart-base')
# config.update({'num_hidden_layers': 12})
# config.update({'type_vocab_size': 5})
config.update({'num_beams': 5})
config.update({'do_sample': False})
config.update({'decoder_layers': 6})
config.update({'encoder_layers': 6})

pretrain_weight = 'facebook/bart-base'
# pretrain_weight = '/local1/cfyang/output/subpolicy/new_subpolicy/checkpoint-1500'
model = BartForSubpolicyGeneration.from_pretrained(pretrain_weight, config=config)
# output_path = "/local1/cfyang/output/subpolicy/label_9_decoder_4_remove_perturb"
# output_path = "/local1/cfyang/output/subpolicy/label_9_decoder_4_perturb_inst"
output_path = "/local1/cfyang/output/subpolicy/sidestep_subpolicy_3x_inst"
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
    compute_metrics=compute_meta_action_metrics,
    # compute_metrics=None
)

# trainer.evaluate(eval_dataset=unseen_eval_dataset)
trainer.train()

# from pyinstrument import Profiler
# profiler = Profiler()
# profiler.start()
# trainer.train()
# profiler.stop()
# print(profiler.output_text(unicode=True, color=True, show_all=True))

# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
#     train_dataloader = trainer.get_train_dataloader()
#     for (i, inputs) in enumerate(train_dataloader):
#         print(i)


# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
# print(prof.key_averages(group_by_stack_n=5).table(sort_by="cpu_time_total", row_limit=20))



