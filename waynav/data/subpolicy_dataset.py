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

class Subpolicy_Dataset:
    def __init__(self, root_dir):
        '''
            predict_xyz: predict xyz coordinate or polar coordinate
        '''
        self.root_dir = root_dir

        self.traj_list = []
        self.img_fn_list = []

        self.img_feat_list = []
        self.class_name_list = []
        self.subpolicy_list = []

        self.starting_view_list = []
        self.start_class_name_list = []

        print("Loading Dataset")
        subpolicy_dict = json.load(open(os.path.join(root_dir, 'subpolicy_new.json')))
        for n in tqdm(os.listdir(root_dir)):
            if os.path.isdir(os.path.join(root_dir, n)):
                for trial in os.listdir(os.path.join(root_dir, n)):
                    try:
                        traj_data = json.load(open(os.path.join(root_dir, n, trial, 'traj.json')))
                        # traj_x = traj_data['traj']['x']
                        # traj_z = traj_data['traj']['z']
                        actseq, target_act, starting_idx, nth_idx = self.traj_2_actseq(traj_data)
                    except Exception as e:
                        continue
                    subpolicy_list = subpolicy_dict[os.path.join(n, trial)]
                    if (max(nth_idx)+1 != len(subpolicy_list)):
                        continue

                    for nav_point_idx in range(len(traj_data["navigation_point"])):
                        nav_point = traj_data["navigation_point"][nav_point_idx]
                        starting_point = traj_data["navigation_point"][starting_idx[nav_point_idx]]
                        if nav_point == starting_point:
                            nth_subpolicy = nth_idx[nav_point_idx]
                            self.subpolicy_list.append(subpolicy_list[nth_subpolicy])

                            self.traj_list.append(traj_data)
                            self.img_fn_list.append(os.path.join(root_dir.replace('/local1/cfyang', '/data/joey'), n, trial, 'images', str(nav_point).zfill(9) + '.png'))
                            
                            class_name = os.path.join(root_dir.replace('/local1/cfyang', '/data/joey'), n, trial, 'class_name', str(nav_point).zfill(9))
                            self.class_name_list.append(class_name)
                            
                            img_feat = [os.path.join(root_dir.replace('/local1/cfyang', '/data/joey'), n, trial, 'split_images', str(nav_point).zfill(9) + '_' + str(x) + '.png').encode() for x in range(4, 12)]
                            # img_feat = [os.path.join(root_dir, n, trial, 'split_images', str(nav_point).zfill(9) + '_' + str(x) + '.png').encode() for x in range(4, 12)]

                            self.img_feat_list.append(img_feat)
                        
                        # self.target_act.append(target_act[nav_point_idx])
                        # self.actseq_list.append(actseq[starting_idx[nav_point_idx]:nav_point_idx+1])

        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
        # feature_fn = '/blip_feature.lmdb'
        feature_fn = '/objects_features.lmdb'
        self.env = lmdb.open(root_dir + feature_fn, subdir=True,
                             readonly=True, lock=False,
                             readahead=False, meminit=False, map_size=109951162777)
        self.class_name_dict = json.load(open(os.path.join(root_dir, 'class_name.json')))
        self.location_dict = json.load(open(os.path.join(root_dir, 'location.json')))
        self.objarg_dict = json.load(open(os.path.join(root_dir, 'objarg.json')))
        # self.subpolicy_to_int = {
        #     'move forward': 3,
        #     'turn left': 4,
        #     'turn right': 5,
        #     'turn around': 6,
        #     'front left': 7,
        #     'front right': 8,
        #     'step back': 9,
        #     'face left': 10,
        #     'face right': 11,
        # }
        self.subpolicy_to_int = {
            'move forward': 3,
            'turn left': 4,
            'turn right': 5,
            'turn around': 6,
            'side step': 7,
            'step back': 8,
            'face left': 9,
            'face right': 10,
        }
        # self.subpolicy_dict = json.load(open(os.path.join(root_dir, 'subpolicy.json')))
        self.original_len = len(self.img_fn_list)

    def __len__(self):
        return self.original_len * 3

    def __getitem__(self, idx):
        inst_idx = idx // self.original_len
        idx = idx % self.original_len
        img_path = self.img_fn_list[idx]
        traj_data = self.traj_list[idx]
        subpolicy = self.subpolicy_list[idx].lower()
        nav_point = int(img_path.split('images/')[1].split('.')[0])
        nav_idx = traj_data['navigation_point']
        for i in range(len(traj_data['instructions'][0])):
            if i not in nav_idx and i > nav_point:
                interaction_point = i
                break

        class_name_dict = self.class_name_dict[self.class_name_list[idx]]

        resnet_feat = self.img_feat_list[idx]

        # instruction = traj_data['instructions'][nav_point]
        instruction = traj_data['instructions'][inst_idx][nav_point]
        location_inst = traj_data['instructions'][0][nav_point]

        location = self.location_dict[location_inst].lower()
        location = "Target: " + location.strip()
        interaction_instruction = traj_data['instructions'][0][interaction_point]
        objarg = self.objarg_dict[interaction_instruction].strip()
        objarg = objarg.replace("Target", "Object")

        # replace_instruction = instruction.lower()
        # replace_instruction = replace_instruction.replace('turn', '<mask>')
        # replace_instruction = replace_instruction.replace('left', '<mask>')
        # replace_instruction = replace_instruction.replace('right', '<mask>')
        # replace_instruction = replace_instruction.replace('around', '<mask>')
        # replace_instruction = replace_instruction.replace('move forward', '<mask>')
        # replace_instruction = replace_instruction.replace('move', '<mask>')
        input_instruction = instruction.lower()+' </s>'+location+' </s>'+objarg
        # input_instruction = replace_instruction.lower()+' </s>'+location+' </s>'+objarg
        # print(input_instruction)
        # input_instruction = location+' </s>'+objarg
        # print(input_instruction)
        input_ids = self.tokenizer(input_instruction)

        # decoder_input_ids = self.tokenizer('</s>'+subpolicy, add_special_tokens=False)
        # labels = self.tokenizer(subpolicy+'</s>', add_special_tokens=False)

        subpolicy = subpolicy.split(' ')
        decoder_input_ids = [2]
        labels = []
        for i in range(len(subpolicy)//2):
            s = subpolicy[i*2] + ' ' + subpolicy[i*2+1]
            decoder_input_ids.append(self.subpolicy_to_int[s])
            labels.append(self.subpolicy_to_int[s])
        labels.append(2)

        obj_input_ids = 0

        all_img_feats = []
        # For direction
        view_idx_lists = []

        with self.env.begin(write=False) as txn:
            # print(resnet_feat)
            for i in range(4):
                res_feat = txn.get(resnet_feat[i])
                res_feat = np.frombuffer(res_feat, dtype=np.float16).reshape(2048,10,10)
                # res_feat = np.frombuffer(res_feat, dtype=np.float32)
                # res_feat = np.zeros((2048,10,10))
                all_img_feats.append(res_feat)

                res_feat = txn.get(resnet_feat[i+4])
                # res_feat = np.frombuffer(res_feat, dtype=np.float32)
                res_feat = np.frombuffer(res_feat, dtype=np.float16).reshape(2048,10,10)
                all_img_feats.append(res_feat)

                view_idx_lists.append(i+1)
                view_idx_lists.append(i+1)

                # view_step_lists.append(1)
                # view_step_lists.append(1)

            # for i in range(4):
            #     res_feat = txn.get(start_resnet_feat[i])
            #     res_feat = np.frombuffer(res_feat, dtype=np.float16).reshape(2048,10,10)
            #     all_img_feats.append(res_feat)

            #     res_feat = txn.get(start_resnet_feat[i+4])
            #     res_feat = np.frombuffer(res_feat, dtype=np.float16).reshape(2048,10,10)
            #     all_img_feats.append(res_feat)

            #     view_idx_lists.append(i+1)
            #     view_idx_lists.append(i+1)

            #     view_step_lists.append(2)
            #     view_step_lists.append(2)

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
                # view_step_lists += [1] * len(obj_input_id["input_ids"])
                view_idx_lists += [i+1] * len(obj_input_id["input_ids"])
            else:
                for k, v in obj_input_id.items():
                    # Remove the CLS token
                    list_to_add = obj_input_id[k][1:]
                    obj_input_ids[k] += list_to_add

            #     view_step_lists += [1] * (len(obj_input_id["input_ids"])-1)
                view_idx_lists += [i+1] * (len(obj_input_id["input_ids"])-1)

        # Object words for starting point
        # for i in range(4):
        #     combined_obj_list = start_class_name_dict[i]
        #     obj_input_id = self.tokenizer(' '.join(combined_obj_list))
            
        #     for k, v in obj_input_id.items():
        #         # Remove the CLS token
        #         list_to_add = obj_input_id[k][1:]
        #         obj_input_ids[k] += list_to_add

        #     view_step_lists += [2] * (len(obj_input_id["input_ids"])-1)
        #     view_idx_lists += [i+1] * (len(obj_input_id["input_ids"])-1)
        
        # view_idx_lists = torch.LongTensor(view_idx_lists)
        # return input_ids, obj_input_ids, all_img_feats, view_idx_lists, decoder_input_ids, labels
        return input_ids, all_img_feats, view_idx_lists, decoder_input_ids, labels, obj_input_ids

    def traj_2_actseq(self, traj_data):
        traj_x = traj_data['traj']['x']
        traj_z = traj_data['traj']['z']
        nav_point = traj_data["navigation_point"]
        instructions = traj_data["instructions"][0]
        rotation = traj_data['traj']["rotation"]

        actseq = []
        target_act = []
        starting_idx = []
        nth_idx = []
        current_instruction = ""
        nth_action = -1
        last_nav_point = -2
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

            if last_nav_point +1 != nav:
                actseq.append('start')
                current_instruction = inst
                starting_idx.append(i)
                nth_action += 1
            else:
                actseq.append(target_act[i-1])
                starting_idx.append(starting_idx[i-1])

            last_nav_point = nav
            nth_idx.append(nth_action)


        return actseq, target_act, starting_idx, nth_idx

class Subpolicy_Pretrain_Dataset:
    def __init__(self, root_dir):
        '''
            predict_xyz: predict xyz coordinate or polar coordinate
        '''
        self.root_dir = root_dir

        self.traj_list = []
        self.img_fn_list = []

        self.img_feat_list = []
        self.class_name_list = []
        self.subpolicy_list = []

        self.starting_view_list = []
        self.start_class_name_list = []

        self.target_act = []

        print("Loading Dataset")
        subpolicy_dict = json.load(open(os.path.join(root_dir, 'subpolicy.json')))
        for n in tqdm(os.listdir(root_dir)):
            if os.path.isdir(os.path.join(root_dir, n)):
                for trial in os.listdir(os.path.join(root_dir, n)):
                    try:
                        traj_data = json.load(open(os.path.join(root_dir, n, trial, 'traj.json')))
                        # traj_x = traj_data['traj']['x']
                        # traj_z = traj_data['traj']['z']
                        actseq, target_act, starting_idx, nth_idx = self.traj_2_actseq(traj_data)
                    except Exception as e:
                        continue
                    subpolicy_list = subpolicy_dict[os.path.join(n, trial)]
                    if (max(nth_idx)+1 != len(subpolicy_list)):
                        continue

                    for nav_point_idx in range(len(traj_data["navigation_point"])):
                        nav_point = traj_data["navigation_point"][nav_point_idx]
                        starting_point = traj_data["navigation_point"][starting_idx[nav_point_idx]]
                        if nav_point == starting_point:
                            nth_subpolicy = nth_idx[nav_point_idx]
                            self.subpolicy_list.append(subpolicy_list[nth_subpolicy])

                            self.traj_list.append(traj_data)
                            self.img_fn_list.append(os.path.join(root_dir.replace('/local1/cfyang', '/data/joey'), n, trial, 'images', str(nav_point).zfill(9) + '.png'))
                            
                            class_name = os.path.join(root_dir.replace('/local1/cfyang', '/data/joey'), n, trial, 'class_name', str(nav_point).zfill(9))
                            self.class_name_list.append(class_name)
                            
                            img_feat = [os.path.join(root_dir.replace('/local1/cfyang', '/data/joey'), n, trial, 'split_images', str(nav_point).zfill(9) + '_' + str(x) + '.png').encode() for x in range(4, 12)]
                            self.img_feat_list.append(img_feat)
                        
                            self.target_act.append(target_act[nav_point_idx])
                        # self.actseq_list.append(actseq[starting_idx[nav_point_idx]:nav_point_idx+1])

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.env = lmdb.open(root_dir + '/objects_features.lmdb', subdir=True,
                             readonly=True, lock=False,
                             readahead=False, meminit=False, map_size=109951162777)
        self.class_name_dict = json.load(open(os.path.join(root_dir, 'class_name.json')))
        self.location_dict = json.load(open(os.path.join(root_dir, 'location.json')))
        self.subpolicy_to_int = {
            'move forward': 3,
            'turn left': 4,
            'turn right': 5,
            'turn around': 6,
            'front left': 7,
            'front right': 8,
            'step back': 9,
            'face left': 10,
            'face right': 11,
        }
        # self.subpolicy_dict = json.load(open(os.path.join(root_dir, 'subpolicy.json')))

    def __len__(self):
        return len(self.img_fn_list[:])

    def __getitem__(self, idx):
        img_path = self.img_fn_list[idx]
        traj_data = self.traj_list[idx]
        subpolicy = self.subpolicy_list[idx].lower()
        nav_point = int(img_path.split('images/')[1].split('.')[0])

        class_name_dict = self.class_name_dict[self.class_name_list[idx]]

        resnet_feat = self.img_feat_list[idx]

        instruction = traj_data['instructions'][nav_point]
        location = self.location_dict[instruction].lower()
        location = "Target: " + location.strip()
         
        input_ids = self.tokenizer(instruction.lower()+' </s>'+location)

        target_act = self.target_act[idx]

        direction, trg_distance = target_act.split('_')
        if direction == 'left':
            trg_direction = 0
        elif direction == 'front':
            trg_direction = 1
        elif direction == 'right':
            trg_direction = 2
        elif direction == 'back':
            trg_direction = 3
        
        subpolicy = subpolicy.split(' ')
        # decoder_input_ids = [2]
        # labels = []
        # for i in range(len(subpolicy)//2):
        try:
            s = subpolicy[0] + ' ' + subpolicy[1]
            trg_subpolicy = self.subpolicy_to_int[s]
        except:
            trg_subpolicy = 1
        #     decoder_input_ids.append(self.subpolicy_to_int[s])
        #     labels.append(self.subpolicy_to_int[s])
        # labels.append(2)
        if trg_subpolicy == 1:
            trg_direction = 1
        elif trg_subpolicy == 3:
            trg_direction = 1
        elif trg_subpolicy == 4:
            trg_direction = 0
        elif trg_subpolicy == 5:
            trg_direction = 2
        elif trg_subpolicy == 6:
            trg_direction = 3
        elif trg_subpolicy == 7:
            trg_direction = 1
        elif trg_subpolicy == 8:
            trg_direction = 1
        elif trg_subpolicy == 9:
            trg_direction = 3
        elif trg_subpolicy == 10:
            trg_direction = 0
        elif trg_subpolicy == 11:
            trg_direction = 2

        obj_input_ids = 0

        all_img_feats = []
        # For direction
        view_idx_lists = []

        with self.env.begin(write=False) as txn:

            for i in range(4):
                res_feat = txn.get(resnet_feat[i])
                res_feat = np.frombuffer(res_feat, dtype=np.float16).reshape(2048,10,10)
                all_img_feats.append(res_feat)

                res_feat = txn.get(resnet_feat[i+4])
                res_feat = np.frombuffer(res_feat, dtype=np.float16).reshape(2048,10,10)
                all_img_feats.append(res_feat)

                view_idx_lists.append(i+1)
                view_idx_lists.append(i+1)

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
            obj_input_id = self.tokenizer(' '.join(combined_obj_list))
            if i == 0:
                obj_input_ids = obj_input_id
                # view_step_lists += [1] * len(obj_input_id["input_ids"])
                view_idx_lists += [i+1] * len(obj_input_id["input_ids"])
            else:
                for k, v in obj_input_id.items():
                    # Remove the CLS token
                    list_to_add = obj_input_id[k][1:]
                    obj_input_ids[k] += list_to_add

            #     view_step_lists += [1] * (len(obj_input_id["input_ids"])-1)
                view_idx_lists += [i+1] * (len(obj_input_id["input_ids"])-1)

        return input_ids, all_img_feats, view_idx_lists, trg_subpolicy, trg_direction, obj_input_ids

    def traj_2_actseq(self, traj_data):
        traj_x = traj_data['traj']['x']
        traj_z = traj_data['traj']['z']
        nav_point = traj_data["navigation_point"]
        instructions = traj_data["instructions"]
        rotation = traj_data['traj']["rotation"]

        actseq = []
        target_act = []
        starting_idx = []
        nth_idx = []
        current_instruction = ""
        nth_action = -1
        last_nav_point = -2
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

            if last_nav_point +1 != nav:
                actseq.append('start')
                current_instruction = inst
                starting_idx.append(i)
                nth_action += 1
            else:
                actseq.append(target_act[i-1])
                starting_idx.append(starting_idx[i-1])

            last_nav_point = nav
            nth_idx.append(nth_action)

        return actseq, target_act, starting_idx, nth_idx

class Subpolicy_NoImg_Dataset:
    def __init__(self, root_dir):
        '''
            predict_xyz: predict xyz coordinate or polar coordinate
        '''
        self.root_dir = root_dir

        self.traj_list = []
        self.img_fn_list = []

        self.img_feat_list = []
        self.class_name_list = []
        self.subpolicy_list = []

        self.starting_view_list = []
        self.start_class_name_list = []

        print("Loading Dataset")
        subpolicy_dict = json.load(open(os.path.join(root_dir, 'subpolicy.json')))
        for n in tqdm(os.listdir(root_dir)):
            if os.path.isdir(os.path.join(root_dir, n)):
                for trial in os.listdir(os.path.join(root_dir, n)):
                    try:
                        traj_data = json.load(open(os.path.join(root_dir, n, trial, 'traj.json')))
                        # traj_x = traj_data['traj']['x']
                        # traj_z = traj_data['traj']['z']
                        actseq, target_act, starting_idx, nth_idx = self.traj_2_actseq(traj_data)
                    except Exception as e:
                        continue
                    subpolicy_list = subpolicy_dict[os.path.join(n, trial)]
                    if (max(nth_idx)+1 != len(subpolicy_list)):
                        continue

                    for nav_point_idx in range(len(traj_data["navigation_point"])):
                        nav_point = traj_data["navigation_point"][nav_point_idx]
                        starting_point = traj_data["navigation_point"][starting_idx[nav_point_idx]]
                        if nav_point == starting_point:
                            nth_subpolicy = nth_idx[nav_point_idx]
                            self.subpolicy_list.append(subpolicy_list[nth_subpolicy])

                            self.traj_list.append(traj_data)
                            self.img_fn_list.append(os.path.join(root_dir.replace('/local1/cfyang', '/data/joey'), n, trial, 'images', str(nav_point).zfill(9) + '.png'))
                            
                            class_name = os.path.join(root_dir.replace('/local1/cfyang/', ''), n, trial, 'class_name', str(nav_point).zfill(9))
                            self.class_name_list.append(class_name)
                            
                            img_feat = [os.path.join(root_dir.replace('/local1/cfyang', '/data/joey'), n, trial, 'split_images', str(nav_point).zfill(9) + '_' + str(x) + '.png').encode() for x in range(4, 12)]
                            self.img_feat_list.append(img_feat)
                        
                        # self.target_act.append(target_act[nav_point_idx])
                        # self.actseq_list.append(actseq[starting_idx[nav_point_idx]:nav_point_idx+1])

        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
        # self.env = lmdb.open(root_dir + '/objects_features.lmdb', subdir=True,
        #                      readonly=True, lock=False,
        #                      readahead=False, meminit=False, map_size=109951162777)

        self.class_name_dict = json.load(open(os.path.join(root_dir, 'class_bbox_depth.json')))
        self.location_dict = json.load(open(os.path.join(root_dir, 'location.json')))
        self.objarg_dict = json.load(open(os.path.join(root_dir, 'objarg.json')))
        self.subpolicy_to_int = {
            'move forward': 3,
            'turn left': 4,
            'turn right': 5,
            'turn around': 6,
            'front left': 7,
            'front right': 8,
            'step back': 9,
            'face left': 10,
            'face right': 11,
        }
        # self.subpolicy_dict = json.load(open(os.path.join(root_dir, 'subpolicy.json')))

    def __len__(self):
        return len(self.img_fn_list)

    def __getitem__(self, idx):
        img_path = self.img_fn_list[idx]
        traj_data = self.traj_list[idx]
        subpolicy = self.subpolicy_list[idx].lower()
        nav_point = int(img_path.split('images/')[1].split('.')[0])
        nav_idx = traj_data['navigation_point']
        for i in range(len(traj_data['instructions'])):
            if i not in nav_idx and i > nav_point:
                interaction_point = i
                break

        class_bbox_depth_dict = self.class_name_dict[self.class_name_list[idx]]

        # resnet_feat = self.img_feat_list[idx]

        instruction = traj_data['instructions'][nav_point]
        location = self.location_dict[instruction].lower()
        location = "Target: " + location.strip()
        interaction_instruction = traj_data['instructions'][interaction_point]
        objarg = self.objarg_dict[interaction_instruction].strip()
        objarg = objarg.replace("Target", "Object")

        input_instruction = instruction.lower()+' </s>'+location+' </s>'+objarg
        input_ids = self.tokenizer(input_instruction)

        # natural language meta action
        # decoder_input_ids = self.tokenizer('</s>'+subpolicy, add_special_tokens=False)
        # labels = self.tokenizer(subpolicy+'</s>', add_special_tokens=False)

        # integer meta action
        subpolicy = subpolicy.split(' ')
        decoder_input_ids = [2]
        labels = []
        for i in range(len(subpolicy)//2):
            s = subpolicy[i*2] + ' ' + subpolicy[i*2+1]
            decoder_input_ids.append(self.subpolicy_to_int[s])
            labels.append(self.subpolicy_to_int[s])
        labels.append(2)

        obj_input_ids = 0
        obj_bbox_lists = [[0,0,0,0]]
        obj_depth_lists = [0]
        view_idx_lists = []
        class_name_dict = class_bbox_depth_dict[0]
        bbox_dict = class_bbox_depth_dict[1]
        depth_dict = class_bbox_depth_dict[2]
        # Object words for current point
        for i in range(4):
            # if i == trg_direction:
            combined_obj_list = ['<s>'] + class_name_dict[i]
            # combined_obj_list = [x.lower() for x in combined_obj_list]
            # combined_obj_list = ' '.join(combined_obj_list)
            combined_bbox_list = bbox_dict[i]
            combined_bbox_list = [item for sublist in combined_bbox_list for item in sublist]
            combined_bbox_list.append([0,0,0,0])
            # print(combined_bbox_list)
            obj_bbox_lists += combined_bbox_list
            combined_depth_list = depth_dict[i]
            combined_depth_list = [item for sublist in combined_depth_list for item in sublist]
            combined_depth_list.append(0)
            obj_depth_lists += combined_depth_list

            # combined_obj_list = '</s>'
            obj_input_id = self.tokenizer(' '.join(combined_obj_list))
            # print(len(obj_input_id['input_ids']), [self.tokenizer.decode(x) for x in obj_input_id['input_ids']])
            if i == 0:
                obj_input_ids = obj_input_id
                for k, v in obj_input_ids.items():
                    # Remove the CLS token
                    obj_input_ids[k] = v[1:]
                view_idx_lists += [i+1] * (len(obj_input_id["input_ids"]))
            else:
                for k, v in obj_input_id.items():
                    # Remove the CLS token
                    list_to_add = obj_input_id[k][2:]
                    obj_input_ids[k] += list_to_add

            #     view_step_lists += [1] * (len(obj_input_id["input_ids"])-1)
                view_idx_lists += [i+1] * (len(obj_input_id["input_ids"])-2)

        # assert len(obj_input_ids['input_ids']) == len(obj_bbox_lists), print(len(obj_input_ids['input_ids']) , len(obj_bbox_lists))
        # print(len(obj_bbox_lists))

        # Object words for starting point
        # for i in range(4):
        #     combined_obj_list = start_class_name_dict[i]
        #     obj_input_id = self.tokenizer(' '.join(combined_obj_list))
            
        #     for k, v in obj_input_id.items():
        #         # Remove the CLS token
        #         list_to_add = obj_input_id[k][1:]
        #         obj_input_ids[k] += list_to_add

        #     view_step_lists += [2] * (len(obj_input_id["input_ids"])-1)
        #     view_idx_lists += [i+1] * (len(obj_input_id["input_ids"])-1)
        
        # view_idx_lists = torch.LongTensor(view_idx_lists)
        # return input_ids, obj_input_ids, all_img_feats, view_idx_lists, decoder_input_ids, labels
        return input_ids, obj_bbox_lists, obj_depth_lists, view_idx_lists, decoder_input_ids, labels, obj_input_ids

    def traj_2_actseq(self, traj_data):
        traj_x = traj_data['traj']['x']
        traj_z = traj_data['traj']['z']
        nav_point = traj_data["navigation_point"]
        instructions = traj_data["instructions"]
        rotation = traj_data['traj']["rotation"]

        actseq = []
        target_act = []
        starting_idx = []
        nth_idx = []
        current_instruction = ""
        nth_action = -1
        last_nav_point = -2
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

            if last_nav_point +1 != nav:
                actseq.append('start')
                current_instruction = inst
                starting_idx.append(i)
                nth_action += 1
            else:
                actseq.append(target_act[i-1])
                starting_idx.append(starting_idx[i-1])

            last_nav_point = nav
            nth_idx.append(nth_action)


        return actseq, target_act, starting_idx, nth_idx
