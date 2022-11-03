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


class Concat_Dataset:
    def __init__(self, root_dir, single_view=True):
        '''
            predict_xyz: predict xyz coordinate or polar coordinate
        '''
        self.root_dir = root_dir

        self.traj_list = []
        self.img_fn_list = []
        self.feat_obj_list = []
        self.feat_recep_list = []
        self.img_feat_len = 15
        for n in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, n)):
                for trial in os.listdir(os.path.join(root_dir, n)):
                    try:
                        traj_data = json.load(open(os.path.join(root_dir, n, trial, 'traj.json')))
                        traj_x = traj_data['traj']['x']
                        traj_z = traj_data['traj']['z']
                    except:
                        continue
                    for nav_point in traj_data["navigation_point"]:
                        # x_diff = traj_x[nav_point+1] - traj_x[nav_point]
                        # z_diff = traj_z[nav_point+1] - traj_z[nav_point]
                        # if x_diff == 0 and z_diff == 0:
                        #     continue
                        self.traj_list.append(traj_data)
                        self.img_fn_list.append(os.path.join(root_dir, n, trial, 'images', str(nav_point).zfill(9) + '.png'))
                        obj_feat = [os.path.join(root_dir, n, trial, 'objects', str(nav_point).zfill(9) + '_' + str(x) + '.npz') for x in range(4, 12)]
                        recep_feat = [os.path.join(root_dir, n, trial, 'objects', str(nav_point).zfill(9) + '_' + str(x) + '.npz') for x in range(4, 12)]
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
        extracted_objects = traj_data['extracted_objects'][nav_point]

        img_feat = []
        input_ids = []
        obj_lists = []
        all_bbox_feats = []
        view_idx_lists = []

        for i in range(8):
            # obj_list_before = (list(obj_feat[i]['pred_class']) + list(recep_feat[i]['pred_class']))
            obj_cls = list(obj_feat[i]['pred_class'])
            recep_cls = list(recep_feat[i]['pred_class'])
            
            # input_ids.append(self.tokenizer(instruction, ' '.join(obj_list)))
            if len(obj_feat[i]['pred_feat'].shape) > 1:
                temp_bbox = obj_feat[i]['pred_box']
                temp_bbox[:,[1,3]] += 300*(i//4)
                temp_bbox /= 300
                obj_bbox_feat = np.concatenate((obj_feat[i]['pred_feat'], temp_bbox), axis=1)
            if len(recep_feat[i]['pred_feat'].shape) > 1:
                temp_bbox = recep_feat[i]['pred_box']
                temp_bbox[:,[1,3]] += 300*(i//4)
                temp_bbox /= 300
                recep_bbox_feat = np.concatenate((recep_feat[i]['pred_feat'], temp_bbox), axis=1)
            
            obj_list = []
            obj_idx = []
            recep_idx = []
            for j in range(len(obj_cls)):
                obj = obj_cls[j]
                # print(obj.lower(), extracted_objects, instruction)
                if obj.lower() in extracted_objects:
                    obj_list.append(obj)
                    obj_idx.append(j)
            for j in range(len(recep_cls)):
                obj = recep_cls[j]
                # print(obj.lower(), extracted_objects, instruction)
                if obj.lower() in extracted_objects:
                    obj_list.append(obj)
                    recep_idx.append(j)
            # print('Before', obj_bbox_feat.shape, recep_bbox_feat.shape)

            obj_bbox_feat = obj_bbox_feat[obj_idx]
            recep_bbox_feat = recep_bbox_feat[recep_idx]
            # print('After', obj_bbox_feat.shape, recep_bbox_feat.shape)
            
            if obj_bbox_feat.shape[0] == 0:
                if recep_bbox_feat.shape[0] == 0:
                    all_bbox_feat = np.zeros((1, 1024+4), dtype=np.float32)
                    obj_list = ['dummy']
                else:
                    all_bbox_feat = recep_bbox_feat
            else:
                if recep_bbox_feat.shape[0] == 0:
                    all_bbox_feat = obj_bbox_feat
                else:
                    all_bbox_feat = np.concatenate((obj_bbox_feat, recep_bbox_feat), axis=0)

            view_idx = [i%4] * len(all_bbox_feat)
            
            obj_lists += (obj_list)
            all_bbox_feats.append(all_bbox_feat)
            view_idx_lists += view_idx
        
        combined_obj_list = list(set(obj_lists))
        input_ids = self.tokenizer(instruction, ' '.join(combined_obj_list))
        combined_bbox_feat = np.concatenate(all_bbox_feats, axis=0)
        # print(combined_bbox_feat.shape, extracted_objects)
        view_idx_lists = np.array(view_idx_lists)
        if combined_bbox_feat.shape[0] >= self.img_feat_len:
            combined_bbox_feat = combined_bbox_feat[:self.img_feat_len]
            view_idx_lists = view_idx_lists[:self.img_feat_len]
        else:
            # [(0,1),(0,1)] -> # of padding before & after the n-th axis
            view_idx_lists = np.pad(view_idx_lists, (0, self.img_feat_len-combined_bbox_feat.shape[0]), constant_values=4)
            combined_bbox_feat = np.pad(combined_bbox_feat, [(0, self.img_feat_len-combined_bbox_feat.shape[0]), (0, 0)], constant_values=0)

        img_feat = torch.tensor(combined_bbox_feat)
        view_idx_lists = torch.LongTensor(view_idx_lists)
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

        direction = 0
        if target_x // 0.25 == 0:
            if target_z // 0.25 == 0:
                direction = 0
            elif target_z > 0:
                direction = 1
            elif target_z < 0:
                direction = 5
        elif target_z // 0.25 == 0:
            if target_x > 0:
                direction = 3
            elif target_x < 0:
                direction = 7
        elif target_x > 0:
            if target_z > 0:
                direction = 2
            elif target_z < 0:
                direction = 4
        elif target_x < 0:
            if target_z > 0:
                direction = 8
            elif target_z < 0:
                direction = 6

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

        # For subinstruction the only prediction is the distance along the z-axis
        target_coord = torch.tensor([target_z])

        panorama_rotation = [0,1,2,3]

        # if self.single_view:
        #     input_ids = [input_ids[target_view]]
        #     img_feat = [img_feat[target_view]]
        #     panorama_rotation = [panorama_rotation[target_view]]
            # view_idx_lists = [view_idx_lists[target_view]]

        # return input_ids, img_feat, panorama_rotation, view_idx_lists, target_coord, target_view
        return [input_ids], [img_feat], [panorama_rotation], target_coord, direction, view_idx_lists

