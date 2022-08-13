from torch import nn
from transformers import Trainer
import torch
from tqdm import tqdm

class WaypointTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids, rgb_list, depth_list, meta_dict = inputs
        target_coord = meta_dict['target_coord']
        target_angle = meta_dict['target_angle']
        target_rotation = meta_dict['target_rotation']
        # forward pass
        outputs = model(input_ids, rgb_list, depth_list, meta_dict['panorama_angle'], meta_dict['panorama_rotation'])
        coord_logits, angle_logits, rotation_logits = outputs

        ce_loss = nn.CrossEntropyLoss()
        mse_loss = nn.MSELoss()
        l1_loss = nn.L1Loss()

        coord_loss = l1_loss(coord_logits.view(-1), target_coord.view(-1))
        angle_loss = ce_loss(angle_logits.view(-1, angle_logits.size(-1)), target_angle)
        rotation_loss = ce_loss(rotation_logits.view(-1, rotation_logits.size(-1)), target_rotation)

        loss = coord_loss + angle_loss + rotation_loss

        return (loss, outputs) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        model = self._wrap_model(self.model, training=False, dataloader=eval_dataloader)
        model.eval()

        l1_loss = nn.L1Loss(reduction='sum')
        data_num = 0
        correct_angle = 0
        correct_rotation = 0
        avg_loss = 0

        for i, (inputs) in enumerate(tqdm(eval_dataloader)):
            inputs = self._prepare_inputs(inputs)
            input_ids, rgb_list, depth_list, meta_dict = inputs
            with torch.no_grad():
                target_coord = meta_dict['target_coord']
                target_angle = meta_dict['target_angle']
                target_rotation = meta_dict['target_rotation']

                outputs = model(input_ids, rgb_list, depth_list, meta_dict['panorama_angle'], meta_dict['panorama_rotation'])
                coord_logits, angle_logits, rotation_logits = outputs
                angle_logits = angle_logits.view(-1, angle_logits.size(-1))
                rotation_logits = rotation_logits.view(-1, rotation_logits.size(-1))

                coord_pred = coord_logits.view(-1)
                angle_pred = torch.argmax(angle_logits, dim=-1)
                rotation_pred = torch.argmax(rotation_logits, dim=-1)

                data_num += angle_pred.size(0)
                correct_angle += torch.sum(angle_pred == target_angle).item()
                correct_rotation += torch.sum(rotation_pred == target_rotation).item()

                avg_loss += l1_loss(coord_logits.view(-1), target_coord.view(-1)).item()

        metrics = {
            "coordinate_mse": avg_loss / (data_num * 2),
            "angle_acc": correct_angle / data_num,
            "rotation_acc": correct_rotation / data_num
        }

        self.log(metrics)

        self._memory_tracker.stop_and_update_metrics(metrics)

        return metrics

