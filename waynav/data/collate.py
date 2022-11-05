import torch
import numpy as np

class RGBD_Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.padding = True
        self.return_tensors = "pt"

    def __call__(self, batch):

        input_id, rgb_list, depth_list, \
        panorama_angle, panorama_rotation, target_coord, target_angle, target_rotation = \
        [], [], [], [], [], [], [], []

        for data in batch:
            input_id.append(data[0]) 
            rgb_list.append(data[1]) 
            depth_list.append(data[2])
            panorama_angle.append(data[3])
            panorama_rotation.append(data[4])
            target_coord.append(data[5])
            target_angle.append(data[6])
            target_rotation.append(data[7])

        input_ids = self.tokenizer.pad(
            input_id,
            padding=self.padding,
            return_tensors=self.return_tensors,
        )
        rgb_list = torch.stack(rgb_list, dim=0)
        depth_list = torch.stack(depth_list, dim=0)

        meta_dict = {
            'panorama_angle': torch.stack(panorama_angle, dim=0),
            'panorama_rotation': torch.stack(panorama_rotation, dim=0),
            'target_coord': torch.stack(target_coord, dim=0),
            'target_angle': torch.LongTensor(target_angle),
            'target_rotation': torch.LongTensor(target_rotation)
        }

        return input_ids, rgb_list, depth_list, meta_dict


class ROI_Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.padding = True
        self.return_tensors = "pt"

    def __call__(self, batch):

        input_id, img_feat, panorama_angle, panorama_rotation, target_coord = [], [], [], [], []

        for data in batch:
            input_id += data[0]
            img_feat += data[1]
            panorama_angle += data[2]
            panorama_rotation += data[3]
            target_coord += data[4]

        input_ids = self.tokenizer.pad(
            input_id,
            padding=self.padding,
            return_tensors=self.return_tensors,
        )
        img_feat = torch.stack(img_feat, dim=0)

        input_dict = {
            'input_ids': input_ids,
            'img_feat': img_feat,
            'panorama_angle': torch.LongTensor(panorama_angle).unsqueeze(1),
            'panorama_rotation': torch.LongTensor(panorama_rotation).unsqueeze(1),
            'target_coord': torch.stack(target_coord, dim=0),
        }

        return input_dict

class Pretrain_Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.padding = True
        self.return_tensors = "pt"

    def __call__(self, batch):

        input_id, img_feat, panorama_rotation, target_coord, target_view = [], [], [], [], []
        view_idx = []

        for data in batch:
            input_id += data[0]
            img_feat += data[1]
            panorama_rotation += data[2]
            # view_idx += data[3]
            target_coord += data[3]
            target_view.append(data[4])
            view_idx.append(data[5])

        target_view = torch.LongTensor(target_view)
        # target_view = torch.nn.functional.one_hot(target_view, num_classes=4).flatten()

        input_ids = self.tokenizer.pad(
            input_id,
            padding=self.padding,
            return_tensors=self.return_tensors,
        )
        img_feat = torch.stack(img_feat, dim=0)
        view_idx = torch.stack(view_idx, dim=0)

        input_dict = {
            'input_ids': input_ids,
            'img_feat': img_feat,
            'panorama_rotation': torch.LongTensor(panorama_rotation).unsqueeze(1),
            'view_idx': view_idx,
            'target_coord': torch.stack(target_coord, dim=0),
            'target_view': target_view.unsqueeze(1),
        }

        return input_dict

class Action_Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.padding = True
        self.return_tensors = "pt"

    def __call__(self, batch):

        input_id, all_img_feats, view_idx_lists, actseq_list, \
        distance_list, timestep_list, trg_direction, trg_distance = [], [], [], [], [], [], [], []

        for data in batch:
            input_id.append(data[0])
            all_img_feats.append(data[1])
            view_idx_lists.append(data[2])
            actseq_list.append(data[3])
            distance_list.append(data[4])
            timestep_list.append(data[5])
            trg_direction.append(data[6])
            trg_distance.append(data[7])

        trg_direction = torch.LongTensor(trg_direction)
        trg_distance = torch.tensor(trg_distance)
        # target_view = torch.nn.functional.one_hot(target_view, num_classes=4).flatten()
        input_ids = self.tokenizer.pad(
            input_id,
            padding=self.padding,
            return_tensors=self.return_tensors,
        )
        actseq_list = self.tokenizer.pad(
            {"input_ids": actseq_list},
            padding=self.padding,
            return_tensors=self.return_tensors,
        )
        distance_list = self.tokenizer.pad(
            {"input_ids": distance_list},
            padding=self.padding,
            return_tensors=self.return_tensors,
        )
        timestep_list = self.tokenizer.pad(
            {"input_ids": timestep_list},
            padding=self.padding,
            return_tensors=self.return_tensors,
        )
        # img_feat = torch.stack(all_img_feats, dim=0)
        # view_idx = torch.stack(view_idx_lists, dim=0)

        img_feat = torch.tensor(np.array(all_img_feats)).float()
        view_idx = torch.tensor(np.array(view_idx_lists)).long()
        img_feat_attn_mask = torch.ones(len(batch), 8).long()
        all_attn_mask = torch.cat((input_ids['attention_mask'], img_feat_attn_mask, actseq_list['attention_mask']),dim=1)

        input_dict = {
            'input_ids': input_ids,
            'img_feat': img_feat,
            'view_idx': view_idx,
            'act_seq': actseq_list['input_ids'],
            'act_dist': distance_list['input_ids'],
            'act_step': timestep_list['input_ids'],
            'target_view': trg_direction,
            'target_coord': trg_distance,
            'attention_mask': all_attn_mask,
        }

        return input_dict