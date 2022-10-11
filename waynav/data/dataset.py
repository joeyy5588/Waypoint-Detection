import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import AutoTokenizer
from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import math
import os
import json
import numpy as np

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _to_tensor(image, mode):
    image = image.convert(mode)
    image = ToTensor()(image)
    return image

def azimuthAngle(x1,  y1,  x2,  y2):
    angle = 0.0
    dx = x2 - x1
    dy = y2 - y1
    if  x2 == x1:
        angle = math.pi / 2.0
        if  y2 == y1 :
            angle = 0.0
        elif y2 < y1 :
            angle = 3.0 * math.pi / 2.0
    elif x2 > x1 and y2 > y1:
        angle = math.atan(dx / dy)
    elif  x2 > x1 and  y2 < y1 :
        angle = math.pi / 2 + math.atan(-dy / dx)
    elif  x2 < x1 and y2 < y1 :
        angle = math.pi + math.atan(dx / dy)
    elif  x2 < x1 and y2 > y1 :
        angle = 3.0 * math.pi / 2.0 + math.atan(dy / -dx)
    return (angle * 180 / math.pi)


class Panorama_Dataset:
    def __init__(self, root_dir, predict_xyz=True):
        '''
            predict_xyz: predict xyz coordinate or polar coordinate
        '''
        self.root_dir = root_dir
        self.img_transform = Compose([
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        self.depth_transform = Compose([
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
        ])

        self.traj_list = []
        self.img_fn_list = []
        for n in os.listdir(root_dir):
            for trial in os.listdir(os.path.join(root_dir, n)):
                try:
                    traj_data = json.load(open(os.path.join(root_dir, n, trial, 'traj.json')))
                except:
                    continue
                for nav_point in traj_data["navigation_point"]:
                    self.traj_list.append(traj_data)
                    self.img_fn_list.append(os.path.join(root_dir, n, trial, 'images', str(nav_point).zfill(9) + '.png'))

        self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-medium")

        self.predict_xyz = predict_xyz

    def __len__(self):
        return len(self.img_fn_list)

    def __getitem__(self, idx):
        img_path = self.img_fn_list[idx]
        depth_path = img_path.replace('images', 'depth')
        traj_data = self.traj_list[idx]
        nav_point = int(img_path.split('images/')[1].split('.')[0])

        # Discretize Traj data
        input_x = traj_data['traj']['x'][nav_point]
        input_z = traj_data['traj']['z'][nav_point]
        input_angle = round(traj_data['traj']['angle'][nav_point]) 
        input_angle = (input_angle + 90) // 15
        input_rotation = round(traj_data['traj']['rotation'][nav_point])
        instruction = traj_data['instructions'][nav_point]
        input_id = self.tokenizer(instruction)# ['input_ids']

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

        # delta_angle = azimuthAngle(0,0,target_x, target_z)
        delta_length = math.sqrt(target_x ** 2 + target_z ** 2)

        # delta_angle = round(traj_data['traj']['angle'][nav_point+1]) - input_angle
        target_angle = round(traj_data['traj']['angle'][nav_point+1])
        target_angle = (target_angle + 90) // 15

        delta_rotation = round(traj_data['traj']['rotation'][nav_point+1]) - input_rotation
        input_rotation = input_rotation // 90
        target_rotation = (delta_rotation % 360) // 90

        rgb_img = Image.open(img_path)
        rgb_img = _to_tensor(rgb_img, "RGB")
        rgb_list = []
        for col in range(4):
            for row in range(3):
                rgb_list.append(rgb_img[:, :300*(row+1), :300*(col+1)])

        depth_img = Image.open(depth_path)
        depth_img = _to_tensor(depth_img, "L") # * 5 / 255
        depth_list = []
        for col in range(4):
            for row in range(3):
                depth_list.append(depth_img[:, :300*(row+1), :300*(col+1)])

        if self.img_transform:
            for img_ind in range(len(rgb_list)):
                rgb_list[img_ind] = self.img_transform(rgb_list[img_ind])

        rgb_list = torch.stack(rgb_list, dim=0)

        if self.depth_transform:
            for img_ind in range(len(depth_list)):
                depth_list[img_ind] = self.depth_transform(depth_list[img_ind])

        depth_list = torch.stack(depth_list, dim=0)

        input_coord = torch.tensor([input_x, input_z])

        # if self.predict_xyz:
        #     target_coord = torch.tensor([target_x, target_z])
        # else:
        #     target_coord = torch.tensor([delta_length, delta_angle])
        target_coord = torch.tensor([target_x, target_z])

        # 30, 0, -30
        panorama_angle = torch.LongTensor([8,8,8,8,6,6,6,6,4,4,4,4])
        # left, center, right, back
        panorama_rotation = torch.LongTensor([3,0,1,2,3,0,1,2,3,0,1,2])

        return input_id, rgb_list, depth_list, \
        panorama_angle, panorama_rotation, target_coord, target_angle, target_rotation

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


class ROI_Dataset:
    def __init__(self, root_dir, predict_xyz=True):
        '''
            predict_xyz: predict xyz coordinate or polar coordinate
        '''
        self.root_dir = root_dir

        self.traj_list = []
        self.img_fn_list = []
        self.feat_obj_list = []
        self.feat_recep_list = []
        self.img_feat_len = 20
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

        self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-medium")

        self.predict_xyz = predict_xyz

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
        for i in range(12):
            obj_list = list(obj_feat[i]['pred_class']) + list(recep_feat[i]['pred_class'])
            input_ids.append(self.tokenizer(instruction, ' '.join(obj_list)))
            if len(obj_feat[i]['pred_feat'].shape) > 1:
                obj_bbox_feat = np.concatenate((obj_feat[i]['pred_feat'], obj_feat[i]['pred_box']/300), axis=1)
            if len(recep_feat[i]['pred_feat'].shape) > 1:
                recep_bbox_feat = np.concatenate((recep_feat[i]['pred_feat'], recep_feat[i]['pred_box']/300), axis=1)
            
            if len(obj_feat[i]['pred_feat'].shape) == 1:
                if len(recep_feat[i]['pred_feat'].shape) == 1:
                    all_bbox_feat = np.zeros((1, 1024+4))
                else:
                    all_bbox_feat = recep_bbox_feat
            else:
                if len(recep_feat[i]['pred_feat'].shape) == 1:
                    all_bbox_feat = obj_bbox_feat
                else:
                    all_bbox_feat = np.concatenate((obj_bbox_feat, recep_bbox_feat), axis=0)

            if all_bbox_feat.shape[0] >= self.img_feat_len:
                all_bbox_feat = all_bbox_feat[:self.img_feat_len]
            else:
                all_bbox_feat = np.pad(all_bbox_feat, [(0, self.img_feat_len-all_bbox_feat.shape[0]), (0, 0)], constant_values=0)
            
            img_feat.append(torch.tensor(all_bbox_feat))
            

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

        input_coord = torch.tensor([input_x, input_z])
        target_coord = []
        for i in range(12):
            target_coord.append(torch.tensor([target_x, target_z]))

        # 30, 0, -30
        panorama_angle = [8,8,8,8,6,6,6,6,4,4,4,4]
        panorama_rotation = [3,0,1,2,3,0,1,2,3,0,1,2]

        return input_ids, img_feat, panorama_angle, panorama_rotation, target_coord

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


