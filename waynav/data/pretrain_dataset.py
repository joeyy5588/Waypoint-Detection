import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

import math
import os
import json
import numpy as np

def azimuthAngle(x1,  y1,  x2,  y2):
    angle = 0.0
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        if dy == 0:
            angle = 0.0
        elif dy > 0:
            angle = 0.0
        else:
            angle = 180.0
    elif dy == 0:
        if dx == 0:
            angle = 0.0
        elif dx > 0:
            angle = 90.0
        else:
            angle = 270.0
    else:
        angle = math.atan2(dx, dy) * 180 / math.pi
        if angle < 0:
            angle += 360

    return angle


class Pretrain_Dataset:
    def __init__(self, root_dir, single_view=True):
        '''
            predict_xyz: predict xyz coordinate or polar coordinate
        '''
        self.root_dir = root_dir

        self.traj_list = []
        self.img_fn_list = []
        self.feat_obj_list = []
        self.feat_recep_list = []
        self.img_feat_len = 30
        for n in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, n)):
                for trial in os.listdir(os.path.join(root_dir, n)):
                    try:
                        traj_data = json.load(open(os.path.join(root_dir, n, trial, 'traj.json')))
                    except:
                        continue
                    for nav_point in traj_data["navigation_point"]:
                        self.traj_list.append(traj_data)
                        self.img_fn_list.append(os.path.join(root_dir, n, trial, 'images', str(nav_point).zfill(9) + '.png'))
                        obj_feat = [os.path.join(root_dir, n, trial, 'objects', str(nav_point).zfill(9) + '_' + str(x) + '.npz') for x in range(12)]
                        recep_feat = [os.path.join(root_dir, n, trial, 'objects', str(nav_point).zfill(9) + '_' + str(x) + '.npz') for x in range(12)]
                        self.feat_obj_list.append(obj_feat)
                        self.feat_recep_list.append(recep_feat)

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        self.single_view = single_view

    def __len__(self):
        return len(self.img_fn_list)

    def __getitem__(self, idx):
        img_path = self.img_fn_list[idx]
        traj_data = self.traj_list[idx]
        nav_point = int(img_path.split('images/')[1].split('.')[0])
        obj_feat = self.feat_obj_list[idx]
        recep_feat = self.feat_recep_list[idx]
        obj_feat = [np.load(x) for x in obj_feat]
        recep_feat = [np.load(x) for x in recep_feat]

        # Discretize Traj data
        input_x = traj_data['traj']['x'][nav_point]
        input_z = traj_data['traj']['z'][nav_point]
        input_angle = round(traj_data['traj']['angle'][nav_point]) 
        input_angle = (input_angle + 90) // 15
        input_rotation = round(traj_data['traj']['rotation'][nav_point])
        instruction = traj_data['instructions'][nav_point]

        img_feat = []
        textual_inputs = []
        input_ids = []
        obj_lists = []
        all_bbox_feats = []

        for i in range(4, 12):
            obj_list = list(obj_feat[i]['pred_class']) + list(recep_feat[i]['pred_class'])
            obj_lists.append(obj_list)
            # input_ids.append(self.tokenizer(instruction, ' '.join(obj_list)))
            if len(obj_feat[i]['pred_feat'].shape) > 1:
                obj_feat[i]['pred_box'][:,[1,3]] += 300*((i-4)//4)
                obj_bbox_feat = np.concatenate((obj_feat[i]['pred_feat'], obj_feat[i]['pred_box']), axis=1)
            if len(recep_feat[i]['pred_feat'].shape) > 1:
                obj_feat[i]['pred_box'][:,[1,3]] += 300*((i-4)//4)
                recep_bbox_feat = np.concatenate((recep_feat[i]['pred_feat'], recep_feat[i]['pred_box']), axis=1)
            
            if len(obj_feat[i]['pred_feat']) == 0:
                if len(recep_feat[i]['pred_feat']) == 0:
                    all_bbox_feat = np.zeros((1, 1024+4), dtype=np.float32)
                else:
                    all_bbox_feat = recep_bbox_feat
            else:
                if len(recep_feat[i]['pred_feat']) == 0:
                    all_bbox_feat = obj_bbox_feat
                else:
                    all_bbox_feat = np.concatenate((obj_bbox_feat, recep_bbox_feat), axis=0)
            
            all_bbox_feats.append(all_bbox_feat)

        for i in range(4):
            combined_obj_list = obj_lists[i] + obj_lists[i+4]# + obj_lists[i+8]
            combined_obj_list = list(set(combined_obj_list))
            input_ids.append(self.tokenizer(instruction, ' '.join(combined_obj_list)))

            combined_bbox_feat = np.concatenate((all_bbox_feats[i], all_bbox_feats[i+4]), axis=0)
            if combined_bbox_feat.shape[0] >= self.img_feat_len:
                combined_bbox_feat = combined_bbox_feat[:self.img_feat_len]
            else:
                # [(0,1),(0,1)] -> # of padding before & after the n-th axis
                combined_bbox_feat = np.pad(combined_bbox_feat, [(0, self.img_feat_len-combined_bbox_feat.shape[0]), (0, 0)], constant_values=0)
            
            img_feat.append(torch.tensor(combined_bbox_feat))

        delta_x = traj_data['traj']['x'][nav_point+1] - traj_data['traj']['x'][nav_point]
        delta_z = traj_data['traj']['z'][nav_point+1] - traj_data['traj']['z'][nav_point]

        if input_rotation == 0:
            target_x = delta_x
            target_z = delta_z
        elif input_rotation == 90:
            target_x = -delta_z
            target_z = delta_x
        elif input_rotation == 180:
            target_x = -delta_x
            target_z = -delta_z
        elif input_rotation == 270:
            target_x = delta_z
            target_z = -delta_x

        delta_angle = azimuthAngle(0, 0, target_x, target_z)

        # Default face to the front, left->front->right->back
        target_view = 1
        if delta_angle <= 45 or delta_angle >= 315:
            target_view = 1
        elif delta_angle <= 135:
            target_view = 2
        elif delta_angle <= 225:
            target_view = 3
        else:
            target_view = 0

        if target_view == 1:
            pass
        elif target_view == 2:
            target_x = -target_z
            target_z = target_x
        elif target_view == 3:
            target_x = -target_x
            target_z = -target_z
        else:
            target_x = target_z
            target_z = -target_x

        target_coord = torch.tensor([target_x, target_z])

        panorama_rotation = [0,1,2,3]

        if self.single_view:
            input_ids = [input_ids[target_view]]
            img_feat = [img_feat[target_view]]
            panorama_rotation = [panorama_rotation[target_view]]

        return input_ids, img_feat, panorama_rotation, target_coord, target_view

    @classmethod
    def process_test_time_data(cls, tokenizer, rgb, depth, instruction):
        input_id = tokenizer(instruction)
        input_ids = tokenizer.pad(
            [input_id],
            padding=True,
            return_tensors='pt',
        )

        img_transform = Compose([
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        depth_transform = Compose([
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
        ])

        rgb_img = ToTensor()(rgb)
        rgb_list = []
        for col in range(4):
            for row in range(3):
                rgb_list.append(rgb_img[:, :300*(row+1), :300*(col+1)])

        depth_img = np.uint8(depth)
        depth_img = ToTensor()(depth_img)
        depth_list = []
        for col in range(4):
            for row in range(3):
                depth_list.append(depth_img[:, :300*(row+1), :300*(col+1)])

        for img_ind in range(len(rgb_list)):
            rgb_list[img_ind] = img_transform(rgb_list[img_ind])

        rgb_list = torch.stack(rgb_list, dim=0).unsqueeze(0)

        for img_ind in range(len(depth_list)):
            depth_list[img_ind] = depth_transform(depth_list[img_ind])

        depth_list = torch.stack(depth_list, dim=0).unsqueeze(0)

        panorama_angle = torch.LongTensor([8,8,8,8,6,6,6,6,4,4,4,4]).unsqueeze(0)
        # left, center, right, back
        panorama_rotation = torch.LongTensor([3,0,1,2,3,0,1,2,3,0,1,2]).unsqueeze(0)

        return input_ids, rgb_list, depth_list, panorama_angle, panorama_rotation


