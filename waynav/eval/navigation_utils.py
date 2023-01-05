import numpy as np
import torch

def prepare_direction_input(tokenizer, patch_feat, obj_cls, recep_cls, instruction, actseq_list, dist_list):

    input_ids = tokenizer(instruction)

    all_img_feats = []
    # For direction
    view_idx_lists = []
    # For starting/current point
    view_step_lists = []

    for i in range(8):
        view_idx_lists.append(i%4 + 1)
        view_step_lists.append(1)

    obj_input_ids = 0
    for i in range(4):
        combined_obj_list = list(obj_cls[i]) + list(obj_cls[i+4]) + list(recep_cls[i])+ list(recep_cls[i+4])
        combined_obj_list = list(set(combined_obj_list))
        obj_input_id = tokenizer(' '.join(combined_obj_list))

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

    timestep_list = list(range(1, len(actseq_list)+1))
    all_attn_mask = torch.ones(len(input_ids['input_ids'])+len(obj_input_ids['input_ids'])+8+len(actseq_list))
    # print(len(obj_input_ids['input_ids']), len(view_idx_lists), len(view_step_lists))

    input_dict = {
            'input_ids': torch.LongTensor(input_ids['input_ids']),
            'obj_input_ids': torch.LongTensor(obj_input_ids['input_ids']),
            'img_feat': patch_feat,
            'view_idx': torch.LongTensor(view_idx_lists),
            'view_step': torch.LongTensor(view_step_lists),
            'act_seq': torch.LongTensor(actseq_list),
            'act_dist': torch.tensor(dist_list),
            'act_step': torch.LongTensor(timestep_list),
            'attention_mask': all_attn_mask,
        }
    return input_dict

def prepare_distance_input(tokenizer, obj_feat, obj_box, obj_cls, recep_feat, recep_box, recep_cls, instruction, target_view):
    obj_lists = []
    all_bbox_feats = []

    for i in range(8):
        if i % 4 == target_view:
            obj_list = list(obj_cls[i]) + list(recep_cls[i])
            single_obj_feat = obj_feat[i].cpu().numpy()
            single_recep_feat = recep_feat[i].cpu().numpy()
            
            # input_ids.append(self.tokenizer(instruction, ' '.join(obj_list)))
            if (single_obj_feat.shape[0]) > 0:
                temp_bbox = obj_box[i]
                temp_bbox[:,[1,3]] += 300*(i//4)
                temp_bbox /= 300
                obj_bbox_feat = np.concatenate((single_obj_feat, temp_bbox), axis=1)
            if (single_recep_feat.shape[0]) > 0:
                temp_bbox = recep_box[i]
                temp_bbox[:,[1,3]] += 300*(i//4)
                temp_bbox /= 300
                recep_bbox_feat = np.concatenate((single_recep_feat, temp_bbox), axis=1)
            
            if (single_obj_feat.shape[0]) == 0:
                if (single_recep_feat.shape[0]) == 0:
                    all_bbox_feat = np.zeros((1, 1024+4), dtype=np.float32)
                    obj_list = ['dummy']
                else:
                    all_bbox_feat = recep_bbox_feat
            else:
                if (single_recep_feat.shape[0]) == 0:
                    all_bbox_feat = obj_bbox_feat
                else:
                    all_bbox_feat = np.concatenate((obj_bbox_feat, recep_bbox_feat), axis=0)
            
            obj_lists.append(obj_list)
            all_bbox_feats.append(all_bbox_feat)

    combined_obj_list = obj_lists[0] + obj_lists[1]# + obj_lists[i+8]
    combined_obj_list = list(set(combined_obj_list))
    input_ids = (tokenizer(instruction, ' '.join(combined_obj_list)))

    combined_bbox_feat = np.concatenate((all_bbox_feats[0], all_bbox_feats[1]), axis=0)

    input_dict = {
        'input_ids': input_ids,
        'img_feat': torch.tensor(combined_bbox_feat),
        'panorama_rotation': torch.LongTensor([target_view]),
    }

    return input_dict

def predict_direction(model, inputs):

    for k, v in inputs.items():
        inputs[k] = v.unsqueeze(0).to(model.device)

    input_ids = inputs['input_ids']
    obj_input_ids = inputs['obj_input_ids']
    img_feat = inputs['img_feat']
    act_seq = inputs['act_seq']
    act_dist = inputs['act_dist']
    act_step = inputs['act_step']
    view_idx = inputs['view_idx']
    view_step = inputs['view_step']
    attention_mask = inputs['attention_mask']

    direction, distance = model(input_ids, obj_input_ids, img_feat, act_seq, act_dist, act_step, view_idx, view_step, attention_mask)
    direction = int(torch.argmax(direction, dim=1).cpu())
    distance = torch.round(distance / 0.25)
    distance = int(distance.cpu())
    distance = max(0, min(distance, 40)) 
    return direction, distance

def predict_distance(model, inputs):
    for k, v in inputs.items():
        if k == 'input_ids':
            v['input_ids'] = torch.LongTensor(v['input_ids']).unsqueeze(0).to(model.device)
            v['token_type_ids'] = torch.LongTensor(v['token_type_ids']).unsqueeze(0).to(model.device)
        else:
            inputs[k] = v.unsqueeze(0).to(model.device)

    input_ids = inputs['input_ids']
    img_feat = inputs['img_feat']
    panorama_rotation = inputs['panorama_rotation']

    distance = model(input_ids, img_feat, panorama_rotation)
    distance = torch.round(distance / 0.25)
    distance = int(distance.cpu())
    distance = max(0, min(distance, 40))

    return distance