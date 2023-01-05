import torch
from waynav.data import Subpolicy_Dataset
from waynav.data.collate import Subpolicy_Collator
from waynav.model import BartForSubpolicyGeneration
from waynav.trainer import SubpolicyTrainer
from transformers import Seq2SeqTrainingArguments, AutoConfig
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np

train_data_dir = '/local1/cfyang/substep_train'
eval_data_dir = '/local1/cfyang/substep_valid_seen'
unseen_eval_data_dir = '/local1/cfyang/substep_valid_unseen'
train_dataset = Subpolicy_Dataset(train_data_dir)
eval_dataset = Subpolicy_Dataset(eval_data_dir)
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
config.update({'num_beams': 1})
config.update({'do_sample': False})


model = BartForSubpolicyGeneration(config).from_pretrained('facebook/bart-base')
output_path = "/local1/cfyang/output/subpolicy/with_object_words"
batch_size = 64
learning_rate = 1e-4
save_steps = 1000

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    ignore_index = (labels == -100)
    if isinstance(preds, tuple):
        preds = np.argmax(preds[0], axis=2)
    # print(preds)
    data_num = preds.shape[0]
    all_equal = np.sum(np.all((preds == labels)|ignore_index, axis=1))
    first_equal = np.sum(np.all(preds[:,:2] == labels[:,:2], axis=1))
    return {"All_Equal": all_equal/data_num, "First_Equal": first_equal/data_num}



training_args = Seq2SeqTrainingArguments(
    output_dir=output_path,
    evaluation_strategy="epoch",
    learning_rate=learning_rate,
    warmup_ratio=0.1,
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
    dataloader_drop_last=True,
    predict_with_generate=False,
    generation_max_length=36,
)

trainer = SubpolicyTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset_dict,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    # compute_metrics=None
)

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



