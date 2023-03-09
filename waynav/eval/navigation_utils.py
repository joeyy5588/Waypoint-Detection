import numpy as np
import torch
from transformers import AutoTokenizer, AutoConfig
from waynav.model import BartForSubpolicyGeneration, VLN_LL_Action, VLN_Boundary
import json, os

class Navigation_Helper:
    def __init__(self, args, device, level):
        # For subpolicy
        if level == 'high':
            config = AutoConfig.from_pretrained('facebook/bart-base')
            config.update({'num_beams': 5})
            self.model = BartForSubpolicyGeneration.from_pretrained(args.subpolicy_model_path, config=config)
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
        # For ll action
        elif level == 'low':
            config = AutoConfig.from_pretrained('bert-base-uncased')
            self.model = VLN_Boundary.from_pretrained(args.ll_model_path, config=config)
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        self.level = level

        split = args.data_path.split('/')[-1]
        self.location_dict = json.load(open(os.path.join(args.metadata_path, split, 'location.json')))
        self.objarg_dict = json.load(open(os.path.join(args.metadata_path, split, 'objarg.json')))
        inst_dict = json.load(open(args.metadata_path+split+'/inst_dict.json'))
        self.inst2type = inst_dict['type']
        
        self.model.to(device)
        self.device = device
        self.model.eval()

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

        self.int_to_subpolicy = {
            3: 'move forward',
            4: 'turn left',
            5: 'turn right',
            6: 'turn around',
            7: 'step left',
            8: 'step right',
            9: 'step back',
            10: 'face left',
            11: 'face right',
        }

        self.int_to_action = {
            0: 'MoveAhead',
            # 1: 'RotateLeft',
            # 2: 'RotateRight',
            # 3: 'Interaction',
            1: 'Interaction',
        }

    def to_device(self, device):
        self.model.to(device)

    def process_instruction(self, instruction, traj_data):
        all_instructions = traj_data['turk_annotations']['anns']
        instruction_ridx = 0
        instruction_high_idx = 0
        for i in range(len(all_instructions)):
            if instruction in all_instructions[i]['high_descs']:
                instruction_ridx = i
                instruction_high_idx = all_instructions[i]['high_descs'].index(instruction)
                break

        location_instruction = all_instructions[0]['high_descs'][instruction_high_idx]

        for i in range(instruction_high_idx, len(all_instructions[instruction_ridx]['high_descs'])):
            if self.inst2type[all_instructions[instruction_ridx]['high_descs'][i].lower().strip()] == 'interaction':
                interaction_instruction = all_instructions[instruction_ridx]['high_descs'][i]
                break
        
        try:
            location = self.location_dict[location_instruction].lower()
        except:
            print('location', img_path, instruction, location_instruction)
            location=''

        location = "Target: " + location.strip()

        try:
            objarg = self.objarg_dict[interaction_instruction].strip()
        except:
            print('objarg', img_path, instruction, interaction_instruction)
            objarg=''
            
        objarg = objarg.replace("Target", "Object")

        if self.level == 'high':
            return instruction.lower()+' </s>'+location+' </s>'+objarg
        else:
            return instruction.lower()+' and '+interaction_instruction.lower() + ' [SEP] ' + location+' [SEP] '+objarg

        return instruction

    def prepare_subpolicy_inputs(self, instruction, traj_data, patch_feat, obj_cls, recep_cls):
        processed_instruction = self.process_instruction(instruction, traj_data)
        input_ids = self.tokenizer(processed_instruction)
        view_idx_lists = []
        for i in range(8):
            view_idx_lists.append(i%4 + 1)

        obj_input_ids = 0
        for i in range(4):
            combined_obj_list = list(obj_cls[i]) + list(obj_cls[i+4]) + list(recep_cls[i])+ list(recep_cls[i+4])
            old_combined_obj_list = list(set(combined_obj_list))
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
                view_idx_lists += [i+1] * len(obj_input_id["input_ids"])

            else:
                for k, v in obj_input_id.items():
                    # Remove the CLS token
                    list_to_add = obj_input_id[k][1:]
                    obj_input_ids[k] += list_to_add

                view_idx_lists += [i+1] * (len(obj_input_id["input_ids"])-1)

        all_attn_mask = torch.ones(len(input_ids['input_ids'])+len(view_idx_lists), dtype=torch.long)
        decoder_input_ids = [2]
        decoder_attention_mask = [1]
        # print(len(obj_input_ids['input_ids']), len(view_idx_lists), len(view_step_lists))

        input_dict = {
            'input_ids': torch.LongTensor(input_ids['input_ids']),
            'obj_input_ids': torch.LongTensor(obj_input_ids['input_ids']),
            'img_feat': patch_feat,
            'view_idx': torch.LongTensor(view_idx_lists),
            'attention_mask': all_attn_mask,
            'decoder_input_ids': torch.LongTensor(decoder_input_ids),
            'decoder_attention_mask': torch.LongTensor(decoder_attention_mask),
        }

        return input_dict

    def prepare_ll_inputs(self, instruction, traj_data, patch_feat, obj_cls, recep_cls, curr_subpolicy, next_subpolicy):
        processed_instruction = self.process_instruction(instruction, traj_data)
        processed_instruction = processed_instruction + ' [SEP] ' + 'current subpolicy: ' + str(curr_subpolicy) + ' [SEP] ' + 'next subpolicy: ' + str(next_subpolicy)
        input_ids = self.tokenizer(processed_instruction)
        view_idx_lists = []
        # view_step_lists = []
        for i in range(8):
            view_idx_lists.append(i%4 + 1)
            # view_step_lists.append(1)

        obj_input_ids = 0
        for i in range(4):
            combined_obj_list = list(obj_cls[i]) + list(obj_cls[i+4]) + list(recep_cls[i])+ list(recep_cls[i+4])
            old_combined_obj_list = list(set(combined_obj_list))
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
                view_idx_lists += [i+1] * len(obj_input_id["input_ids"])
                # view_step_lists += [1] * len(obj_input_id["input_ids"])
            else:
                for k, v in obj_input_id.items():
                    # Remove the CLS token
                    list_to_add = obj_input_id[k][1:]
                    obj_input_ids[k] += list_to_add

                view_idx_lists += [i+1] * (len(obj_input_id["input_ids"])-1)
                # view_step_lists += [1] * (len(obj_input_id["input_ids"])-1)

        all_attn_mask = torch.ones(len(input_ids['input_ids'])+len(view_idx_lists), dtype=torch.long)

        input_dict = {
            'input_ids': torch.LongTensor(input_ids['input_ids']),
            'obj_input_ids': torch.LongTensor(obj_input_ids['input_ids']),
            'img_feat': patch_feat,
            'view_idx': torch.LongTensor(view_idx_lists),
            # 'view_step': torch.LongTensor(view_step_lists),
            'attention_mask': all_attn_mask,
            # 'subpolicy': torch.LongTensor([subpolicy]),
        }

        return input_dict

    def predict(self, inst, traj_data, patch_feat, obj_cls, recep_cls, curr_subpolicy=None, next_subpolicy=None):
        if self.level == 'high':
            inputs = self.prepare_subpolicy_inputs(inst, traj_data, patch_feat, obj_cls, recep_cls)
            for k, v in inputs.items():
                inputs[k] = v.unsqueeze(0).to(self.device)

            gen_kwargs = {}
            gen_kwargs["max_length"] = self.model.config.max_length
            gen_kwargs["num_beams"] = 5
            gen_kwargs["synced_gpus"] = False

            if "attention_mask" in inputs:
                gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)

            if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
                generation_inputs = inputs[self.model.encoder.main_input_name]
            else:
                generation_inputs = inputs[self.model.main_input_name]

            irrelevant_prefix = ["decoder_", "cross_attn", "use_cache", "input_ids"]
            for argument, value in inputs.items():
                if not any(argument.startswith(p) for p in irrelevant_prefix):
                    gen_kwargs[argument] = value

            with torch.no_grad():

                generated_tokens = self.model.generate(
                    generation_inputs,
                    **gen_kwargs,
                )
                preds = generated_tokens.cpu().tolist()[0][1:]
                preds = preds[:preds.index(2)]
                preds = [self.int_to_subpolicy[p] for p in preds]
                return preds
        else:
            inputs = self.prepare_ll_inputs(inst, traj_data, patch_feat, obj_cls, recep_cls, curr_subpolicy, next_subpolicy)
            for k, v in inputs.items():
                inputs[k] = v.unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                pred_action = outputs
                pred_action = torch.argmax(pred_action, dim=-1).cpu().tolist()[0]
                pred_action = self.int_to_action[pred_action]
                return pred_action
        return

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

