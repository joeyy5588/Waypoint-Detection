import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from waynav.gen import constants

import math
import os
import json
import numpy as np
from tqdm import tqdm
import lmdb
from random import shuffle

class Low_Level_Dataset:
    def __init__(self, root_dir):
        '''
            predict_xyz: predict xyz coordinate or polar coordinate
        '''
        self.root_dir = root_dir
        self.traj_list = []
        self.img_fn_list = []

        self.img_feat_list = []
        self.class_name_list = []
        self.start_img_feat_list = []
        self.start_class_name_list = []
        self.subpolicy_list = []
        self.action_list = []

        self.subpolicy_dict = json.load(open(os.path.join(root_dir, 'll_subpolicy_sidestep.json')))

        self.boundary_list = []
        self.left_list = []
        self.move_list = []
        self.right_list = []
        invalid_list = json.load(open(os.path.join(root_dir, 'invalid_list.json')))
        print("Loading Dataset")
        for n in tqdm(os.listdir(root_dir)):
            if os.path.isdir(os.path.join(root_dir, n)):
                for trial in os.listdir(os.path.join(root_dir, n)):
                    if trial in invalid_list:
                        continue
                    try:
                        traj_data = json.load(open(os.path.join(root_dir, n, trial, 'traj.json')))
                        # traj_x = traj_data['traj']['x']
                        # traj_z = traj_data['traj']['z']
                        subpolicy_list = self.subpolicy_dict[os.path.join(n, trial)]
                        starting_idx, nth_subpolicy, subpolicy_bound = self.traj_2_actseq(traj_data, subpolicy_list)
                        do_interaction = traj_data['do_interaction']
                        ll_action_list = traj_data['ll_action_list']
                        # print(traj_data['ll_action_list'])
                        # print(subpolicy_list)
                        # print(subpolicy_bound)

                    except Exception as e:
                        # print(e)
                        continue

                    for img_idx, img_fn in enumerate(sorted(os.listdir(os.path.join(root_dir, n, trial, 'images')))):
                        if do_interaction[img_idx] == 1:
                            if do_interaction[img_idx-1] == 1:
                                continue
                        if subpolicy_bound[img_idx] == 1:
                            continue
                        elif subpolicy_bound[img_idx] == 2:
                            subpolicy = subpolicy_list[img_idx-1]
                            action = 'boundary'
                        else:
                            subpolicy = subpolicy_list[img_idx]
                            action = ll_action_list[img_idx]
                        if action == 'boundary':
                            self.boundary_list.append(len(self.img_fn_list))
                        elif action == 'MoveAhead':
                            self.move_list.append(len(self.img_fn_list))
                        elif action == 'RotateLeft':
                            self.left_list.append(len(self.img_fn_list))
                        elif action == 'RotateRight':
                            self.right_list.append(len(self.img_fn_list))
                        self.action_list.append(action)
                        self.traj_list.append(traj_data)
                        self.subpolicy_list.append(subpolicy)
                        self.img_fn_list.append(os.path.join(root_dir, n, trial, 'images', img_fn))
                        self.class_name_list.append(os.path.join(root_dir, n, trial, 'class_name', str(img_idx).zfill(9)))
                        img_feat = [os.path.join(root_dir, n, trial, 'objectsfeatures', str(img_idx).zfill(9) + '_' + str(x) + '.png') for x in range(8)]
                        self.img_feat_list.append(img_feat)
                        self.start_class_name_list.append(os.path.join(root_dir, n, trial, 'class_name', str(starting_idx[img_idx]).zfill(9)))
                        self.start_img_feat_list.append([os.path.join(root_dir, n, trial, 'objectsfeatures', str(starting_idx[img_idx]).zfill(9) + '_' + str(x) + '.png') for x in range(8)])
        
        # print(len(self.boundary_list), len(self.left_list), len(self.right_list), len(self.move_list))
        shuffle(self.move_list)
        self.all_img_list = self.boundary_list + self.left_list*7+self.right_list*7+self.move_list[:45000]
        shuffle(self.all_img_list)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        # feature_fn = '/blip_feature.lmdb'
        feature_fn = '/objects_features.lmdb'
        self.env = lmdb.open(root_dir + feature_fn, subdir=True,
                             readonly=True, lock=False,
                             readahead=False, meminit=False, map_size=109951162777)
        self.class_name_dict = json.load(open(os.path.join(root_dir, 'class_name.json')))
        self.location_dict = json.load(open(os.path.join(root_dir, 'location.json')))
        self.objarg_dict = json.load(open(os.path.join(root_dir, 'objarg.json')))

        self.ll_to_int = {
            'MoveAhead': 0,
            'RotateLeft': 1,
            'RotateRight': 2,
        }
        self.subpolicy_to_int = {
            'move forward': 0,
            'turn left': 1,
            'turn right': 2,
            'turn around': 3,
            'step left': 4,
            'step right': 5,
            'step back': 6,
            'face left': 7,
            'face right': 8,
        }
        # self.subpolicy_to_int = {
        #     'move forward': 0,
        #     'turn left': 1,
        #     'turn right': 2,
        #     'turn around': 3,
        #     'side step': 4,
        #     'step back': 5,
        #     'face left': 6,
        #     'face right': 7,
        # }

    def traj_2_actseq(self, traj_data, subpolicy_list):
        instructions = traj_data["instructions"]
        do_interaction = traj_data["do_interaction"]
        starting_idx = []
        nth_subpolicy = []
        current_instruction = ""
        curr_pointer = 0
        subpolicy_pointer = 0
        subpolicy_bound = []
        for i, ins in enumerate(instructions):
            if ins != current_instruction:
                curr_pointer = i
                current_instruction = ins
            if i > 0:
                if do_interaction[i] != do_interaction[i-1] and do_interaction[i] == 1:
                    subpolicy_pointer += 1
            starting_idx.append(curr_pointer)
            nth_subpolicy.append(subpolicy_pointer)

        curr_sub = subpolicy_list[0]
        # 1 means discard data, 2 means its subpolicy should be previous one
        for i, sub in enumerate(subpolicy_list):
            if i == 0:
                if 'turn' in sub or 'interaction' in sub:
                    subpolicy_bound.append(1)
                else:
                    subpolicy_bound.append(0)
            elif 'turn' in sub:
                if ('interaction' not in subpolicy_list[i-1]) and (sub != subpolicy_list[i-1]):
                    subpolicy_bound.append(2)
                else:
                    subpolicy_bound.append(1)
            elif sub != curr_sub:
                if ('turn' in subpolicy_list[i-1]) or ('interaction' in subpolicy_list[i-1]):
                    if ('interaction' in subpolicy_list[i]):
                        subpolicy_bound.append(1)
                    else:
                        subpolicy_bound.append(0)
                else:
                    subpolicy_bound.append(2)
            else:
                subpolicy_bound.append(0)

            curr_sub = sub
                
        return starting_idx, nth_subpolicy, subpolicy_bound

    def __len__(self):
        return len(self.all_img_list)

    def __getitem__(self, idx):
        idx = self.all_img_list[idx]
        img_path = self.img_fn_list[idx]
        traj_data = self.traj_list[idx]
        nav_point = int(img_path.split('images/')[1].split('.')[0])
        ll_action = traj_data['ll_action_list'][nav_point]
        if ll_action not in self.ll_to_int:
            nav_point -= 1
        class_name_dict = self.class_name_dict[self.class_name_list[idx]]
        subpolicy = self.subpolicy_list[idx]
        subpolicy = self.subpolicy_to_int[subpolicy]
        action = self.action_list[idx]
        resnet_feat = self.img_feat_list[idx]

        start_resnet_feat = self.start_img_feat_list[idx]
        start_class_name_dict = self.class_name_dict[self.start_class_name_list[idx]]

        instruction = traj_data['instructions'][nav_point]
        for i in range(nav_point, len(traj_data['instructions'])):
            if traj_data['do_interaction'][i] == 1:
                interaction_point = i
                break
        interaction_instruction = traj_data['instructions'][interaction_point]
        location = self.location_dict[instruction].lower()
        location = "Target: " + location.strip()
        objarg = self.objarg_dict[interaction_instruction].strip()
        objarg = objarg.replace("Target: ", "Object:")
        input_instruction = instruction.lower()+' and '+interaction_instruction.lower() + ' [SEP] ' + location+' [SEP] '+objarg

        if action not in self.ll_to_int:
            labels = 3
        else:
            labels = self.ll_to_int[action]
        # print(input_instruction, labels)
        # input_ids = self.tokenizer(input_instruction)

        obj_input_ids = 0

        all_img_feats = []
        # For direction
        view_idx_lists = []
        view_step_lists = []

        for i in range(4):
            res_feat = np.load(resnet_feat[i].replace('.png', '.npz.npy'))
            all_img_feats.append(res_feat)

            res_feat = np.load(resnet_feat[i+4].replace('.png', '.npz.npy'))
            all_img_feats.append(res_feat)

            view_idx_lists.append(i+1)
            view_idx_lists.append(i+1)
            view_step_lists.append(1)
            view_step_lists.append(1)
        
        for i in range(4):
            res_feat = np.load(start_resnet_feat[i].replace('.png', '.npz.npy'))
            all_img_feats.append(res_feat)

            res_feat = np.load(start_resnet_feat[i+4].replace('.png', '.npz.npy'))
            all_img_feats.append(res_feat)

            view_idx_lists.append(i+1)
            view_idx_lists.append(i+1)
            view_step_lists.append(2)
            view_step_lists.append(2)

        all_img_feats = np.array(all_img_feats)


        # Object words for current point
        for i in range(4):
            # if i == trg_direction:
            old_combined_obj_list = class_name_dict[i]
            combined_obj_list = []
            for cobj in old_combined_obj_list:
                if cobj.lower() in instruction:
                    combined_obj_list.append(cobj)
                elif 'table' in cobj.lower():
                    combined_obj_list.append(cobj)
            # combined_obj_list = '</s>'
            obj_input_id = self.tokenizer(' '.join(combined_obj_list))
            if i == 0:
                obj_input_ids = obj_input_id
                view_step_lists += [1] * len(obj_input_id["input_ids"])
                view_idx_lists += [i+1] * len(obj_input_id["input_ids"])
            else:
                for k, v in obj_input_id.items():
                    # Remove the CLS token
                    list_to_add = obj_input_id[k][1:]
                    obj_input_ids[k] += list_to_add

                view_step_lists += [1] * (len(obj_input_id["input_ids"])-1)
                view_idx_lists += [i+1] * (len(obj_input_id["input_ids"])-1)

        for i in range(4):
            # if i == trg_direction:
            old_combined_obj_list = start_class_name_dict[i]
            combined_obj_list = []
            for cobj in old_combined_obj_list:
                if cobj.lower() in instruction:
                    combined_obj_list.append(cobj)
                elif 'table' in cobj.lower():
                    combined_obj_list.append(cobj)
            # combined_obj_list = '</s>'
            obj_input_id = self.tokenizer(' '.join(combined_obj_list))
            for k, v in obj_input_id.items():
                # Remove the CLS token
                list_to_add = obj_input_id[k][1:]
                obj_input_ids[k] += list_to_add

            view_step_lists += [2] * (len(obj_input_id["input_ids"])-1)
            view_idx_lists += [i+1] * (len(obj_input_id["input_ids"])-1)

        
        # view_idx_lists = torch.LongTensor(view_idx_lists)
        # print(input_ids, all_img_feats, view_idx_lists, label, interaction_label, obj_input_ids)
        return input_instruction, all_img_feats, view_idx_lists, labels, subpolicy, obj_input_ids, view_step_lists