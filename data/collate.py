import torch

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
        rgb_list = torch.cat(rgb_list, dim=0)
        depth_list = torch.cat(depth_list, dim=0)

        meta_dict = {
            'panorama_angle': torch.stack(panorama_angle, dim=0),
            'panorama_rotation': torch.stack(panorama_rotation, dim=0),
            'target_coord': torch.stack(target_coord, dim=0),
            'target_angle': torch.LongTensor(target_angle),
            'target_rotation': torch.LongTensor(target_rotation)
        }

        return input_ids, rgb_list, depth_list, meta_dict