from torch import nn
from transformers import Seq2SeqTrainer
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm
from waynav.trainer.loss import SILogLoss
from transformers.debug_utils import DebugOption
from transformers.deepspeed import deepspeed_init
from transformers.utils import is_sagemaker_mp_enabled, logging, is_torch_tpu_available
from transformers.trainer_utils import TrainOutput, has_length, speed_metrics
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.trainer_callback import TrainerState
import torch.distributed as dist
import math, time, os
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.deepspeed import deepspeed_config, is_deepspeed_zero3_enabled


logger = logging.get_logger(__name__)

class SubpolicyTrainer(Seq2SeqTrainer):
    def __init__(self,
        model = None,
        args = None,
        data_collator = None,
        train_dataset = None,
        eval_dataset = None,
        tokenizer = None,
        model_init = None,
        compute_metrics = None,
        callbacks = None,
        optimizers = (None, None),
        preprocess_logits_for_metrics = None):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer,\
        model_init, compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)

        self.softmax = nn.Softmax(dim=-1)

    # def training_step(self, model, inputs):
    #     model.train()
    #     inputs = self._prepare_inputs(inputs)

    #     if is_sagemaker_mp_enabled():
    #         loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
    #         return loss_mb.reduce_mean().detach().to(self.args.device)

    #     with self.compute_loss_context_manager():
    #         loss, loss_metric = self.compute_loss(model, inputs)

    #     if self.args.n_gpu > 1:
    #         loss = loss.mean()  # mean() to average on multi-gpu parallel training

    #     if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
    #         # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
    #         loss = loss / self.args.gradient_accumulation_steps

    #     if self.do_grad_scaling:
    #         self.scaler.scale(loss).backward()
    #     elif self.use_apex:
    #         with amp.scale_loss(loss, self.optimizer) as scaled_loss:
    #             scaled_loss.backward()
    #     elif self.deepspeed:
    #         # loss gets scaled under gradient_accumulation_steps in deepspeed
    #         loss = self.deepspeed.backward(loss)
    #     else:
    #         loss.backward()

    #     return loss.detach(), loss_metric

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     # input_ids = inputs['input_ids']
    #     # obj_input_ids = inputs['obj_input_ids']
    #     # img_feat = inputs['img_feat']
    #     # view_idx = inputs['view_idx']
    #     # attention_mask = inputs['attention_mask']
    #     # decoder_input_ids = inputs['decoder_input_ids']
    #     # labels = inputs['labels']
    #     # forward pass
    #     outputs = model(**inputs)
    #     training_loss = outputs[0]
    #     # Pretrain waypoint predictor/ view selector
    #     loss_metric = {
    #         "training_loss": training_loss.mean().item(),
    #         # "direction_loss": direction_loss.item()
    #     }

    #     return (training_loss, outputs, loss_metric) if return_outputs else (training_loss, loss_metric)

    # def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
    #     self._memory_tracker.start()

    #     eval_dataloader = self.get_eval_dataloader(eval_dataset)
    #     model = self._wrap_model(self.model, training=False, dataloader=eval_dataloader)
    #     model.eval()

    #     ce_loss = nn.CrossEntropyLoss(ignore_index=1)
    #     first_success = 0
    #     entire_success = 0
    #     data_num = 0
    #     for i, (inputs) in enumerate(tqdm(eval_dataloader)):
    #         if (inputs['labels'].size(0)) != self.args.train_batch_size:
    #             continue
    #         inputs = self._prepare_inputs(inputs)
    #         with torch.no_grad():
    #             generation_inputs = {
    #                 'input_ids': inputs['input_ids'],
    #                 'img_feat': inputs['img_feat'],
    #                 'view_idx': inputs['view_idx'],
    #                 'attention_mask': inputs['attention_mask'],
    #             }
    #             encoder = model.get_encoder()
    #             encoder_outputs = encoder(**inputs)
    #             gen_kwargs = self._gen_kwargs
    #             gen_kwargs['encoder_outputs'] = encoder_outputs
    #             generated_tokens = model.generate(
    #                 generation_inputs,
    #                 **gen_kwargs,
    #             )
    #             print(generated_tokens)
    #             data_num += inputs['labels'].size(0)

    #     metrics = {
    #         # f"{metric_key_prefix}_direction_loss": direction_loss / data_num,
    #         # f"{metric_key_prefix}_direction_acc": direction_correct / data_num,
    #         f"{metric_key_prefix}_first_policy_success": distance_loss / data_num,
    #         f"{metric_key_prefix}_entire_success": distance_correct / data_num,
    #     }
    #     print(metrics)

    #     self.log(metrics)

    #     self._memory_tracker.stop_and_update_metrics(metrics)

    #     return metrics

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.
        Subclass and override to inject custom behavior.
        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = self._gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.model.config.max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.model.config.num_beams
        )
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)
        if "global_attention_mask" in inputs:
            gen_kwargs["global_attention_mask"] = inputs.get("global_attention_mask", None)

        # prepare generation inputs
        # some encoder-decoder models can have varying encoder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]
        
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache", "input_ids"]
        for argument, value in inputs.items():
            if not any(argument.startswith(p) for p in irrelevant_prefix):
                gen_kwargs[argument] = value
        

        generated_tokens = self.model.generate(
            generation_inputs,
            **gen_kwargs,
        )
        # print(generated_tokens)
        # in case the batch is shorter than max length, the output should be padded
        if gen_kwargs.get("max_length") is not None and generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])
        elif gen_kwargs.get("max_new_tokens") is not None and generated_tokens.shape[-1] < (
            gen_kwargs["max_new_tokens"] + 1
        ):
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_new_tokens"] + 1)

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if gen_kwargs.get("max_length") is not None and labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
            elif gen_kwargs.get("max_new_tokens") is not None and labels.shape[-1] < (
                gen_kwargs["max_new_tokens"] + 1
            ):
                labels = self._pad_tensors_to_max_len(labels, (gen_kwargs["max_new_tokens"] + 1))
        else:
            labels = None

        return (loss, generated_tokens, labels)

    # def _inner_training_loop(
    #     self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    # ):
    #     self._train_batch_size = batch_size
    #     # Data loader and number of training steps
    #     train_dataloader = self.get_train_dataloader()

    #     # Setting up training control variables:
    #     # number of training epochs: num_train_epochs
    #     # number of training steps per epoch: num_update_steps_per_epoch
    #     # total number of training steps to execute: max_steps
    #     total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

    #     len_dataloader = None
    #     if has_length(train_dataloader):
    #         len_dataloader = len(train_dataloader)
    #         num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
    #         num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    #         num_examples = self.num_examples(train_dataloader)
    #         if args.max_steps > 0:
    #             max_steps = args.max_steps
    #             num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
    #                 args.max_steps % num_update_steps_per_epoch > 0
    #             )
    #             # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
    #             # the best we can do.
    #             num_train_samples = args.max_steps * total_train_batch_size
    #         else:
    #             max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
    #             num_train_epochs = math.ceil(args.num_train_epochs)
    #             num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
    #     elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
    #         max_steps = args.max_steps
    #         # Setting a very large number of epochs so we go as many times as necessary over the iterator.
    #         num_train_epochs = sys.maxsize
    #         num_update_steps_per_epoch = max_steps
    #         num_examples = total_train_batch_size * args.max_steps
    #         num_train_samples = args.max_steps * total_train_batch_size
    #     else:
    #         raise ValueError(
    #             "args.max_steps must be set to a positive value if dataloader does not have a length, was"
    #             f" {args.max_steps}"
    #         )

    #     if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
    #         if self.args.n_gpu > 1:
    #             # nn.DataParallel(model) replicates the model, creating new variables and module
    #             # references registered here no longer work on other gpus, breaking the module
    #             raise ValueError(
    #                 "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
    #                 " (torch.distributed.launch)."
    #             )
    #         else:
    #             debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

    #     delay_optimizer_creation = (
    #         self.sharded_ddp is not None
    #         and self.sharded_ddp != ShardedDDPOption.SIMPLE
    #         or is_sagemaker_mp_enabled()
    #         or self.fsdp is not None
    #     )
    #     if args.deepspeed:
    #         deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
    #             self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
    #         )
    #         self.model = deepspeed_engine.module
    #         self.model_wrapped = deepspeed_engine
    #         self.deepspeed = deepspeed_engine
    #         self.optimizer = optimizer
    #         self.lr_scheduler = lr_scheduler
    #     elif not delay_optimizer_creation:
    #         self.create_optimizer_and_scheduler(num_training_steps=max_steps)

    #     self.state = TrainerState()
    #     self.state.is_hyper_param_search = trial is not None

    #     # Activate gradient checkpointing if needed
    #     if args.gradient_checkpointing:
    #         self.model.gradient_checkpointing_enable()

    #     model = self._wrap_model(self.model_wrapped)

    #     if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
    #         self._load_from_checkpoint(resume_from_checkpoint, model)

    #     # for the rest of this function `model` is the outside model, whether it was wrapped or not
    #     if model is not self.model:
    #         self.model_wrapped = model

    #     if delay_optimizer_creation:
    #         self.create_optimizer_and_scheduler(num_training_steps=max_steps)

    #     # Check if saved optimizer or scheduler states exist
    #     self._load_optimizer_and_scheduler(resume_from_checkpoint)

    #     # important: at this point:
    #     # self.model         is the Transformers Model
    #     # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

    #     # Train!
    #     logger.info("***** Running training *****")
    #     logger.info(f"  Num examples = {num_examples}")
    #     logger.info(f"  Num Epochs = {num_train_epochs}")
    #     logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    #     logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    #     logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    #     logger.info(f"  Total optimization steps = {max_steps}")

    #     self.state.epoch = 0
    #     start_time = time.time()
    #     epochs_trained = 0
    #     steps_trained_in_current_epoch = 0
    #     steps_trained_progress_bar = None

    #     # Check if continuing training from a checkpoint
    #     if resume_from_checkpoint is not None and os.path.isfile(
    #         os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
    #     ):
    #         self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
    #         epochs_trained = self.state.global_step // num_update_steps_per_epoch
    #         if not args.ignore_data_skip:
    #             steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
    #             steps_trained_in_current_epoch *= args.gradient_accumulation_steps
    #         else:
    #             steps_trained_in_current_epoch = 0

    #         logger.info("  Continuing training from checkpoint, will skip to saved global_step")
    #         logger.info(f"  Continuing training from epoch {epochs_trained}")
    #         logger.info(f"  Continuing training from global step {self.state.global_step}")
    #         if not args.ignore_data_skip:
    #             logger.info(
    #                 f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
    #                 "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
    #                 "flag to your launch command, but you will resume the training on data already seen by your model."
    #             )
    #             if self.is_local_process_zero() and not args.disable_tqdm:
    #                 steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
    #                 steps_trained_progress_bar.set_description("Skipping the first batches")

    #     # Update the references
    #     self.callback_handler.model = self.model
    #     self.callback_handler.optimizer = self.optimizer
    #     self.callback_handler.lr_scheduler = self.lr_scheduler
    #     self.callback_handler.train_dataloader = train_dataloader
    #     self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
    #     if trial is not None:
    #         assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
    #         self.state.trial_params = hp_params(assignments)
    #     else:
    #         self.state.trial_params = None
    #     # This should be the same if the state has been saved but in case the training arguments changed, it's safer
    #     # to set this after the load.
    #     self.state.max_steps = max_steps
    #     self.state.num_train_epochs = num_train_epochs
    #     self.state.is_local_process_zero = self.is_local_process_zero()
    #     self.state.is_world_process_zero = self.is_world_process_zero()

    #     # tr_loss is a tensor to avoid synchronization of TPUs through .item()
    #     # loss_metric = {
    #     #     'coord_loss': 0,
    #     #     'angle_loss': 0,
    #     #     'rotation_loss': 0
    #     # }
    #     tr_loss = torch.tensor(0.0).to(args.device)
    #     # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
    #     self._total_loss_scalar = 0.0
    #     self._globalstep_last_logged = self.state.global_step
    #     model.zero_grad()

    #     self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

    #     # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
    #     if not args.ignore_data_skip:
    #         for epoch in range(epochs_trained):
    #             is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
    #                 train_dataloader.sampler, RandomSampler
    #             )
    #             if version.parse(torch.__version__) < version.parse("1.11") or not is_random_sampler:
    #                 # We just need to begin an iteration to create the randomization of the sampler.
    #                 # That was before PyTorch 1.11 however...
    #                 for _ in train_dataloader:
    #                     break
    #             else:
    #                 # Otherwise we need to call the whooooole sampler cause there is some random operation added
    #                 # AT THE VERY END!
    #                 _ = list(train_dataloader.sampler)

    #     for epoch in range(epochs_trained, num_train_epochs):
    #         if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
    #             train_dataloader.sampler.set_epoch(epoch)
    #         elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
    #             train_dataloader.dataset.set_epoch(epoch)

    #         if is_torch_tpu_available():
    #             parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
    #             epoch_iterator = parallel_loader
    #         else:
    #             epoch_iterator = train_dataloader

    #         # Reset the past mems state at the beginning of each epoch if necessary.
    #         if args.past_index >= 0:
    #             self._past = None

    #         steps_in_epoch = (
    #             len(epoch_iterator)
    #             if len_dataloader is not None
    #             else args.max_steps * args.gradient_accumulation_steps
    #         )
    #         self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

    #         if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
    #             self._load_rng_state(resume_from_checkpoint)

    #         step = -1
    #         first_time = 0

    #         for step, inputs in enumerate(epoch_iterator):
    #             # for k, v in inputs.items():
    #             #     try:
    #             #         print(k, v.size())
    #             #     except:
    #             #         print(k, v['input_ids'].size())
    #             if (inputs['labels'].size(0)) != (args.train_batch_size):
    #                 break
    #             # Skip past any already trained steps if resuming training
    #             if steps_trained_in_current_epoch > 0:
    #                 steps_trained_in_current_epoch -= 1
    #                 if steps_trained_progress_bar is not None:
    #                     steps_trained_progress_bar.update(1)
    #                 if steps_trained_in_current_epoch == 0:
    #                     self._load_rng_state(resume_from_checkpoint)
    #                 continue
    #             elif steps_trained_progress_bar is not None:
    #                 steps_trained_progress_bar.close()
    #                 steps_trained_progress_bar = None

    #             if step % args.gradient_accumulation_steps == 0:
    #                 self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

    #             # if (step // 20) % 2 == 0 and step <= 1000:
    #             #     for param in model.module.view_selector.parameters():
    #             #         param.requires_grad = False
    #             #     for param in model.module.waypoint_predictor.parameters():
    #             #         param.requires_grad = True
    #             # else:
    #             #     for param in model.module.waypoint_predictor.parameters():
    #             #         param.requires_grad = False
    #             #     for param in model.module.view_selector.parameters():
    #             #         param.requires_grad = True

    #             if (
    #                 ((step + 1) % args.gradient_accumulation_steps != 0)
    #                 and args.local_rank != -1
    #                 and args._no_sync_in_gradient_accumulation
    #             ):
    #                 # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
    #                 with model.no_sync():
    #                     tr_loss_step, loss_metric = self.training_step(model, inputs)
    #             else:
    #                 tr_loss_step, loss_metric = self.training_step(model, inputs)

    #             if (
    #                 args.logging_nan_inf_filter
    #                 and not is_torch_tpu_available()
    #                 and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
    #             ):
    #                 # if loss is nan or inf simply add the average of previous logged losses
    #                 tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
    #             else:
    #                 tr_loss += tr_loss_step

    #             # self.current_flos += float(self.floating_point_ops(inputs))

    #             # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
    #             if self.deepspeed:
    #                 self.deepspeed.step()

    #             if (step + 1) % args.gradient_accumulation_steps == 0 or (
    #                 # last step in epoch but step is always smaller than gradient_accumulation_steps
    #                 steps_in_epoch <= args.gradient_accumulation_steps
    #                 and (step + 1) == steps_in_epoch
    #             ):
    #                 # Gradient clipping
    #                 if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
    #                     # deepspeed does its own clipping

    #                     if self.do_grad_scaling:
    #                         # Reduce gradients first for XLA
    #                         if is_torch_tpu_available():
    #                             gradients = xm._fetch_gradients(self.optimizer)
    #                             xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
    #                         # AMP: gradients need unscaling
    #                         self.scaler.unscale_(self.optimizer)

    #                     if is_sagemaker_mp_enabled() and args.fp16:
    #                         self.optimizer.clip_master_grads(args.max_grad_norm)
    #                     elif hasattr(self.optimizer, "clip_grad_norm"):
    #                         # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
    #                         self.optimizer.clip_grad_norm(args.max_grad_norm)
    #                     elif hasattr(model, "clip_grad_norm_"):
    #                         # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
    #                         model.clip_grad_norm_(args.max_grad_norm)
    #                     else:
    #                         # Revert to normal clipping otherwise, handling Apex or full precision
    #                         nn.utils.clip_grad_norm_(
    #                             amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
    #                             args.max_grad_norm,
    #                         )

    #                 # Optimizer step
    #                 optimizer_was_run = True
    #                 if self.deepspeed:
    #                     pass  # called outside the loop
    #                 elif is_torch_tpu_available():
    #                     if self.do_grad_scaling:
    #                         self.scaler.step(self.optimizer)
    #                         self.scaler.update()
    #                     else:
    #                         xm.optimizer_step(self.optimizer)
    #                 elif self.do_grad_scaling:
    #                     scale_before = self.scaler.get_scale()
    #                     self.scaler.step(self.optimizer)
    #                     self.scaler.update()
    #                     scale_after = self.scaler.get_scale()
    #                     optimizer_was_run = scale_before <= scale_after
    #                 else:
    #                     self.optimizer.step()

    #                 if optimizer_was_run and not self.deepspeed:
    #                     self.lr_scheduler.step()

    #                 model.zero_grad()
    #                 self.state.global_step += 1
    #                 self.state.epoch = epoch + (step + 1) / steps_in_epoch
    #                 self.control = self.callback_handler.on_step_end(args, self.state, self.control)

    #                 self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval, loss_metric)
    #             else:
    #                 self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

    #             if self.control.should_epoch_stop or self.control.should_training_stop:
    #                 break
    #         if step < 0:
    #             logger.warning(
    #                 "There seems to be not a single sample in your epoch_iterator, stopping training at step"
    #                 f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
    #                 f" num_steps ({max_steps}) higher than the number of available samples."
    #             )
    #             self.control.should_training_stop = True

    #         self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
    #         self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval, loss_metric)

    #         if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
    #             if is_torch_tpu_available():
    #                 # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
    #                 xm.master_print(met.metrics_report())
    #             else:
    #                 logger.warning(
    #                     "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
    #                     "configured. Check your training configuration if this is unexpected."
    #                 )
    #         if self.control.should_training_stop:
    #             break

    #     if args.past_index and hasattr(self, "_past"):
    #         # Clean the state at the end of training
    #         delattr(self, "_past")

    #     logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
    #     if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
    #         # Wait for everyone to get here so we are sur the model has been saved by process 0.
    #         if is_torch_tpu_available():
    #             xm.rendezvous("load_best_model_at_end")
    #         elif args.local_rank != -1:
    #             dist.barrier()
    #         elif is_sagemaker_mp_enabled():
    #             smp.barrier()

    #         self._load_best_model()

    #     # add remaining tr_loss
    #     self._total_loss_scalar += tr_loss.item()
    #     train_loss = self._total_loss_scalar / self.state.global_step

    #     metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
    #     self.store_flos()
    #     metrics["total_flos"] = self.state.total_flos
    #     metrics["train_loss"] = train_loss
    #     for k, v in loss_metric.items():
    #         metrics[k] = v

    #     self.is_in_train = False

    #     self._memory_tracker.stop_and_update_metrics(metrics)

    #     self.log(metrics)

    #     self.control = self.callback_handler.on_train_end(args, self.state, self.control)

    # def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval, loss_metric):
    #     if self.control.should_log:
    #         if is_torch_tpu_available():
    #             xm.mark_step()

    #         logs: Dict[str, float] = {}

    #         # all_gather + mean() to get average loss over all processes
    #         tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

    #         # reset tr_loss to zero
    #         tr_loss -= tr_loss
    #         # for k, v in loss_metric.items():
    #         #     loss_metric[k] -= v

    #         logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
    #         for k, v in loss_metric.items():
    #             logs[k] = round(v, 4)
    #         logs["learning_rate"] = self._get_learning_rate()

    #         self._total_loss_scalar += tr_loss_scalar
    #         self._globalstep_last_logged = self.state.global_step
    #         self.store_flos()

    #         self.log(logs)

    #     metrics = None
    #     if self.control.should_evaluate:
    #         if isinstance(self.eval_dataset, dict):
    #             for eval_dataset_name, eval_dataset in self.eval_dataset.items():
    #                 metrics = self.evaluate(
    #                     eval_dataset=eval_dataset,
    #                     ignore_keys=ignore_keys_for_eval,
    #                     metric_key_prefix=f"eval_{eval_dataset_name}",
    #                 )
    #         else:
    #             metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
    #         self._report_to_hp_search(trial, self.state.global_step, metrics)


    #     if self.control.should_save:
    #         self._save_checkpoint(model, trial, metrics=metrics)
    #         self.control = self.callback_handler.on_save(self.args, self.state, self.control)


    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)