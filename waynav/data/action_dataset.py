import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

import math
import os
import json
import numpy as np
from tqdm import tqdm
import lmdb

class Action_Dataset:
    def __init__(self, root_dir):
        '''
            predict_xyz: predict xyz coordinate or polar coordinate
        '''
        self.root_dir = root_dir

        self.traj_list = []
        self.img_fn_list = []
        self.img_feat_list = []
        self.feat_obj_list = []
        self.class_name_list = []
        self.feat_recep_list = []
        self.actseq_list = []
        self.target_act = []
        self.actseq_len = 20
        print("Loading Dataset")
        for n in tqdm(os.listdir(root_dir)):
            if os.path.isdir(os.path.join(root_dir, n)):
                for trial in os.listdir(os.path.join(root_dir, n)):
                    try:
                        traj_data = json.load(open(os.path.join(root_dir, n, trial, 'traj.json')))
                        traj_x = traj_data['traj']['x']
                        traj_z = traj_data['traj']['z']
                        actseq, target_act, starting_idx = self.traj_2_actseq(traj_data)
                    except Exception as e:
                        continue
                    for nav_point_idx in range(len(traj_data["navigation_point"])):
                        nav_point = traj_data["navigation_point"][nav_point_idx]
                        # x_diff = traj_x[nav_point+1] - traj_x[nav_point]
                        # z_diff = traj_z[nav_point+1] - traj_z[nav_point]
                        # if x_diff == 0 and z_diff == 0:
                        #     continue
                        self.traj_list.append(traj_data)
                        self.img_fn_list.append(os.path.join(root_dir, n, trial, 'images', str(nav_point).zfill(9) + '.png'))
                        # obj_feat = [os.path.join(root_dir, n, trial, 'objects', str(nav_point).zfill(9) + '_' + str(x) + '.npz') for x in range(4, 12)]
                        # recep_feat = [os.path.join(root_dir, n, trial, 'receptacles', str(nav_point).zfill(9) + '_' + str(x) + '.npz') for x in range(4, 12)]
                        class_name = os.path.join(root_dir, n, trial, 'class_name', str(nav_point).zfill(9))
                        self.class_name_list.append(class_name)
                        # self.feat_obj_list.append(obj_feat)
                        # self.feat_recep_list.append(recep_feat)
                        img_feat = [os.path.join(root_dir, n, trial, 'split_images', str(nav_point).zfill(9) + '_' + str(x) + '.png').encode() for x in range(4, 12)]
                        # img_feat = [np.load(x) for x in img_feat]
                        self.img_feat_list.append(img_feat)
                        self.target_act.append(target_act[nav_point_idx])
                        self.actseq_list.append(actseq[starting_idx[nav_point_idx]:nav_point_idx+1])

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.env = lmdb.open(root_dir + '/objects_features.lmdb', subdir=True,
                             readonly=True, lock=False,
                             readahead=False, meminit=False, map_size=109951162777)
        self.class_name_dict = json.load(open(os.path.join(root_dir, 'class_name.json')))

    def __len__(self):
        return len(self.img_fn_list)

    def __getitem__(self, idx):
        img_path = self.img_fn_list[idx]
        traj_data = self.traj_list[idx]
        nav_point = int(img_path.split('images/')[1].split('.')[0])
        class_name_dict = self.class_name_dict[self.class_name_list[idx]]
        # obj_feat = self.feat_obj_list[idx]
        # recep_feat = self.feat_recep_list[idx]
        # obj_feat = [np.load(x) for x in obj_feat]
        # recep_feat = [np.load(x) for x in recep_feat]
        resnet_feat = self.img_feat_list[idx]
        target_act = self.target_act[idx]
        actseq = self.actseq_list[idx]

        instruction = traj_data['instructions'][nav_point]
        input_ids = self.tokenizer(instruction)
        all_img_feats = []
        view_idx_lists = []

        with self.env.begin(write=False) as txn:
            for i in range(4):
                # obj_list = list(obj_feat[i]['pred_class']) + list(recep_feat[i]['pred_class']) \
                # + list(obj_feat[i+4]['pred_class']) + list(recep_feat[i+4]['pred_class'])   
                # combined_obj_list = list(set(obj_list))
                combined_obj_list = class_name_dict[i]
                obj_input_id = self.tokenizer(' '.join(combined_obj_list))
                for k, v in obj_input_id.items():
                    # Remove the CLS token
                    list_to_add = obj_input_id[k][1:]
                    if k == 'token_type_ids':
                        list_to_add = [x+1+i for x in list_to_add]

                    input_ids[k] += list_to_add

                res_feat = txn.get(resnet_feat[i])
                res_feat = np.frombuffer(res_feat, dtype=np.float16).reshape(2048,10,10)
                all_img_feats.append(res_feat)

                res_feat = txn.get(resnet_feat[i+4])
                res_feat = np.frombuffer(res_feat, dtype=np.float16).reshape(2048,10,10)
                all_img_feats.append(res_feat)

                view_idx_lists.append(i+1)
                view_idx_lists.append(i+1)

        all_img_feats = np.array(all_img_feats)

        actseq_list = []
        distance_list = []
        timestep_list = []
        for step, act in enumerate(actseq):
            timestep_list.append(step+1)
            if act == 'start':
                actseq_list.append(1)
                distance = 0
            else:
                direction, distance = act.split('_')
                distance = int(float(distance) // 0.25)
                if direction == 'left':
                    actseq_list.append(2)
                elif direction == 'front':
                    actseq_list.append(3)
                elif direction == 'right':
                    actseq_list.append(4)
                elif direction == 'back':
                    actseq_list.append(5)
            
            distance_list.append(distance)

        direction, trg_distance = target_act.split('_')
        if direction == 'left':
            trg_direction = 0
        elif direction == 'front':
            trg_direction = 1
        elif direction == 'right':
            trg_direction = 2
        elif direction == 'back':
            trg_direction = 3

        trg_distance = float(trg_distance)
        # view_idx_lists = torch.LongTensor(view_idx_lists)
        return input_ids, all_img_feats, view_idx_lists, actseq_list, distance_list, timestep_list, trg_direction, trg_distance

    def traj_2_actseq(self, traj_data):
        traj_x = traj_data['traj']['x']
        traj_z = traj_data['traj']['z']
        nav_point = traj_data["navigation_point"]
        instructions = traj_data["instructions"]
        rotation = traj_data['traj']["rotation"]

        actseq = []
        target_act = []
        starting_idx = []
        current_instruction = ""
        for i in range(len(nav_point)):
            nav = nav_point[i]
            inst = instructions[nav]

            # if nav not in nav_point:
            #     if nav-1 in nav_point:
            #         actseq.append('terminate')
            #         continue
                        
            rot = round(rotation[nav])
            delta_x = traj_x[nav+1] - traj_x[nav]
            delta_z = traj_z[nav+1] - traj_z[nav]

            if rot == 0:
                target_x = delta_x
                target_z = delta_z
            elif rot == 90:
                target_x = -delta_z
                target_z = delta_x
            elif rot == 180:
                target_x = -delta_x
                target_z = -delta_z
            elif rot == 270:
                target_x = delta_z
                target_z = -delta_x

            direction = 0
            if target_x // 0.25 == 0:
                if target_z // 0.25 == 0:
                    delta_rot = round(rotation[nav+1]) - round(rotation[nav])
                    delta_rot  = delta_rot % 360
                    if delta_rot == 0:
                        actstr = 'front_'
                    elif delta_rot == 90:
                        actstr = 'right_'
                    elif delta_rot == 180:
                        actstr = 'back_'
                    elif delta_rot == 270:
                        actstr = 'left_'
                    target_act.append(actstr + str(abs(target_z)))
                elif target_z > 0:
                    target_act.append('front_' + str(abs(target_z)))
                elif target_z < 0:
                    target_act.append('back_' + str(abs(target_z)))
            elif target_z // 0.25 == 0:
                if target_x > 0:
                    target_act.append('right_' + str(abs(target_x)))
                elif target_x < 0:
                    target_act.append('left_' + str(abs(target_x)))

            if inst != current_instruction:
                actseq.append('start')
                current_instruction = inst
                starting_idx.append(i)
            else:
                actseq.append(target_act[i-1])
                starting_idx.append(starting_idx[i-1])


        return actseq, target_act, starting_idx

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


