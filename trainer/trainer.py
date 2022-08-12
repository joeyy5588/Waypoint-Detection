from torch import nn
from transformers import Trainer


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
