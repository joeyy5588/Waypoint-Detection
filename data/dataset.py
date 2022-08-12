import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import os
import json

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _to_tensor(image, mode):
    image = image.convert(mode)
    image = ToTensor()(image)
    return image

class Panorama_Dataset:
    def __init__(self, root_dir):
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

    def __len__(self):
        return len(self.img_fn_list)

    def __getitem__(self, idx):
        img_path = self.img_fn_list[idx]
        depth_path = img_path.replace('images', 'depth')
        traj_data = self.traj_list[idx]
        nav_point = int(img_path.split('images/')[1].split('.')[0])

        input_x = traj_data['traj']['x'][nav_point]
        input_z = traj_data['traj']['z'][nav_point]
        input_angle = round(traj_data['traj']['angle'][nav_point])
        input_rotation = round(traj_data['traj']['rotation'][nav_point])
        instruction = traj_data['instructions'][nav_point]

        target_x = traj_data['traj']['x'][nav_point+1] - traj_data['traj']['x'][nav_point]
        target_z = traj_data['traj']['z'][nav_point+1] - traj_data['traj']['z'][nav_point]
        target_angle = round(traj_data['traj']['angle'][nav_point+1])
        target_rotation = round(traj_data['traj']['rotation'][nav_point+1])

        rgb_img = Image.open(img_path)
        rgb_img = _to_tensor(rgb_img, "RGB")
        rgb_list = []
        for col in range(4):
            for row in range(3):
                rgb_list.append(rgb_img[:, :300*(row+1), :300*(col+1)])

        depth_img = Image.open(depth_path)
        depth_img = _to_tensor(depth_img, "L") * 5 / 255
        depth_list = []
        for col in range(4):
            for row in range(3):
                depth_list.append(depth_img[:, :300*(row+1), :300*(col+1)])

        if self.img_transform:
            for img_ind in range(len(rgb_list)):
                rgb_list[img_ind] = self.img_transform(rgb_list[img_ind])

        if self.depth_transform:
            for img_ind in range(len(depth_list)):
                depth_list[img_ind] = self.img_transform(depth_list[img_ind])

        meta_dict = {
            'input_coord': [input_x, input_z],
            'input_angle': input_angle,
            'input_rotation': input_rotation,
            'instruction': instruction,
            'target_coord': [target_x, target_z],
            'target_angle': target_angle,
            'target_rotation': target_rotation
        }

        return rgb_list, depth_list, meta_dict
