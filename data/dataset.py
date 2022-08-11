import torch
from torch.utils.data import Dataset
import os
import json

class Panorama_Dataset:
    def __init__(self, root_dir, img_transform=None, depth_transform=None):
        self.root_dir = root_dir
        self.img_transform = img_transform
        self.depth_transform = depth_transform

        self.traj_list = []
        self.img_fn_list = []
        for n in os.listdir(root_dir):
            for trial in os.listdir(os.path.join(root_dir, n)):
                traj_data = json.load(open(os.path.join(root_dir, n, trial, 'traj.json')))
                for nav_point in traj_data["navigation_point"]:
                    self.traj_list.append(traj_data)
                    self.img_fn_list.append(os.path.join(root_dir, n, trial, 'images', str(nav_point).zfill(9) + '.png'))

    def __len__(self):
        return len(self.img_fn_list)

    def __getitem__(self, idx):
        img_path = self.img_fn_list[idx]
        depth_path = img_path.replace('images', 'depth')
        traj_data = traj_list[idx]
        nav_point = int(img_path.split('images/')[1].split('.')[0])

        input_x = traj_data['traj']['x'][nav_point]
        input_z = traj_data['traj']['z'][nav_point]
        input_angle = traj_data['traj']['angle'][nav_point]
        input_rotation = traj_data['traj']['rotation'][nav_point]
        instruction = traj_data['instructions'][nav_point]

        target_x = traj_data['traj']['x'][nav_point+1] - traj_data['traj']['x'][nav_point]
        target_z = traj_data['traj']['z'][nav_point+1] - traj_data['traj']['z'][nav_point]
        target_angle = traj_data['traj']['angle'][nav_point+1]
        target_rotation = traj_data['traj']['rotation'][nav_point+1]

        if self.img_transform:
            rgb_img = self.img_transform(rgb_img)

        if self.depth_transform:
            depth_img = self.depth_transform(depth_img)

        meta_dict = {
            'input_x': input_x,
            'input_z': input_z,
            'input_angle': input_angle,
            'input_rotation': input_rotation,
            'instruction': instruction,
            'target_x': target_x,
            'target_z': target_z,
            'target_angle': target_angle,
            'target_rotation': target_rotation
        }
        print(meta_dict)
        return rgb_img, depth_img, meta_dict
