import torch
import pandas as pd
import os
import cv2
import json
import torch
import random
import pickle
import numpy as np
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as tf
from torchvision.io import read_image
from scipy import sparse

colors_map = torch.tensor(
    [   
        [0, 0, 0, 255],  # unknown
        [255, 158, 0, 255],  #  1 car  orange
        [255, 99, 71, 255],  #  2 truck  Tomato
        [255, 140, 0, 255],  #  3 trailer  Darkorange
        [255, 69, 0, 255],  #  4 bus  Orangered
        [233, 150, 70, 255],  #  5 construction_vehicle  Darksalmon
        [220, 20, 60, 255],  #  6 bicycle  Crimson
        [255, 61, 99, 255],  #  7 motorcycle  Red
        [0, 0, 230, 255],  #  8 pedestrian  Blue
        [47, 79, 79, 255],  #  9 traffic_cone  Darkslategrey
        [112, 128, 144, 255],  #  10 barrier  Slategrey
        [0, 207, 191, 255],  # 11  driveable_surface  nuTonomy green  
        [175, 0, 75, 255],  #  12 other_flat  
        [75, 0, 75, 255],  #  13  sidewalk 
        [112, 180, 60, 255],  # 14 terrain  
        [222, 184, 135, 255], # 15 manmade Burlywood 
        [0, 175, 0, 255],  # 16 vegetation  Green
    ]
).type(torch.uint8)


class NuScenesDatasetMtvSpar(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer, trainorval):
        self.args = args
        self.trainorval = trainorval

        dataroot = args.dataroot_path
        self.save_name = args.mtp_path

        if trainorval == 'train':
            data_file = os.path.join(dataroot, 'nuscenes_occ_infos_train.pkl')
        elif trainorval == 'val':
            data_file = os.path.join(dataroot, 'nuscenes_occ_infos_val.pkl')


        with open(data_file, "rb") as file:
            nus_pkl = pickle.load(file)

        self.dataset = nus_pkl['infos']#[:500]
        
        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        # random.shuffle(self.dataset)    

        transforms_list = [
                transforms.Resize((self.args.height, self.args.width), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]

        if trainorval == 'train':
            transforms_list.append(transforms.Normalize([0.5], [0.5]))

        self.image_transforms = transforms.Compose(transforms_list)

        
        self.prompt = ['show a photorealistic street view image']
        # self.CAM_NAMES = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']
        self.CAM_NAMES = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']


        if self.args.use_sdxl:
            self.tokenizer, self.text_encoders = tokenizer
            self.input_ids = self.tokenize_captions_sdxl(self.prompt * 6)
        else:
            self.tokenizer = tokenizer
            self.input_ids = self.tokenize_captions(self.prompt)

        if self.args.use_cbgs and self.trainorval == 'train':
            self.CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                            'barrier', 'background')
            self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
            self.use_valid_flag = True

            self.sample_indices = self._get_sample_indices()
            self.length = len(self.sample_indices)
            print(f"cbgs data scale: {self.length}")


        self.weight_dtype = torch.float16





    def tokenize_captions(self, examples, is_train=True):
        captions = []
        for caption in examples:
            if random.random() < self.args.proportion_empty_prompts:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column should contain either strings or lists of strings."
                )
        inputs = self.tokenizer(
            captions, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    def tokenize_captions_sdxl(self, prompt_batch, is_train=True):

        original_size = (self.args.width, self.args.height)
        target_size = (self.args.width, self.args.height)
        crops_coords_top_left = (0, 0)

        prompt_embeds, pooled_prompt_embeds = self.encode_prompt(
            prompt_batch, self.text_encoders, self.tokenizer, self.args.proportion_empty_prompts, is_train
        )
        add_text_embeds = pooled_prompt_embeds

        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])

        add_time_ids = add_time_ids.repeat(len(prompt_batch), 1)

        return {
            "prompt_ids": prompt_embeds,
            "unet_added_conditions": {"text_embeds": add_text_embeds, "time_ids": add_time_ids},
        }


    # Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
    def encode_prompt(self, prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train=True):
        prompt_embeds_list = []

        captions = []
        for caption in prompt_batch:
            if random.random() < proportion_empty_prompts:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])

        with torch.no_grad():
            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                text_inputs = tokenizer(
                    captions,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids
                prompt_embeds = text_encoder(
                    text_input_ids.to(text_encoder.device),
                    output_hidden_states=True,
                )

                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds.hidden_states[-2]
                bs_embed, seq_len, _ = prompt_embeds.shape
                prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
                prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
        return prompt_embeds, pooled_prompt_embeds



    def __len__(self):
        'Denotes the total number of samples'
        if self.args.use_cbgs and self.trainorval == 'train':
            return len(self.sample_indices)
        return self.length

    def __getitem__(self, idx):
        if self.args.use_cbgs and self.trainorval == 'train':
            idx = self.sample_indices[idx]
        return self.get_data_dict(idx)

    def get_data_dict(self, index):

        curr_info = self.dataset[index]

        mtv_img = []
        mtv_path_img = []
        mtv_condition = []
        mtv_prompt = []
        mtv_weight_mask = []
        mtv_occ_rgb = []

        for cam_id in range(6):

            path_img = curr_info['cams'][self.CAM_NAMES[cam_id]]['data_path']

            img = Image.open(path_img).convert("RGB")
            img = self.image_transforms(img)[None]
            mtv_img.append(img)

            pth_path = path_img.replace('jpg', 'pth')
            all_path = pth_path.replace('samples', self.save_name)


            in_cha = self.args.ctrl_channel - 1
            ctrl_img_path = all_path[:-4] + f'_mtp{in_cha}.npz'
            ctrl_img = sparse.load_npz(ctrl_img_path) 
            ctrl_img = ctrl_img.toarray().reshape((in_cha, self.args.height//8, self.args.width//8))
            ctrl_img = torch.tensor(ctrl_img)[None]#.to  (device='cuda', dtype=self.weight_dtype) #/ 16



            fuse_path = all_path[:-4] + '_fuseweight.png'
            fuse_img = torch.tensor(cv2.imread(fuse_path, cv2.IMREAD_GRAYSCALE))#[..., 0]
            # fuse_img_down = tf.resize(fuse_img[None], (self.args.height//8, self.args.width//8))[0]
            fuse_img_down = fuse_img

            ctrl_img = torch.cat([ctrl_img, fuse_img_down[None, None]], 1)

            mtv_condition.append(ctrl_img)
            mtv_path_img.append(path_img)

            if not self.args.use_sdxl:
                # input_ids = self.tokenize_captions(self.prompt)
                input_ids = self.input_ids
                mtv_prompt.append(input_ids[None])

            if self.trainorval == 'val':
                occrgb_path = all_path[:-4] + '_occrgb.png'
                # occrgb_path = all_path[:-4] + '_occrgb.jpg'
                occ_rgb = read_image(occrgb_path).permute(1,2,0) # 3hw to hw3
                occ_rgb = tf.resize(occ_rgb.permute(2,0,1), (self.args.height, self.args.width)).permute(1,2,0)
                mtv_occ_rgb.append(occ_rgb[None]) # hw3

            elif self.trainorval == 'train':
                mask = (fuse_img_down >= 1) & (fuse_img_down <= 10)
                weight_mask = torch.ones_like(mask) * 1
                weight_mask[mask] = 2

                depth_path = all_path[:-4] + '_depthmap.png'
                depth_map = torch.tensor(cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE))
                # depth_map = tf.resize(depth_map[None], (self.args.height//8, self.args.width//8))[0]
                weight_mask = torch.cat([weight_mask[None], depth_map[None]], 0)
                mtv_weight_mask.append(weight_mask[None])


        mtv_img = torch.cat(mtv_img, 0)
        mtv_condition = torch.cat(mtv_condition, 0)


        data_dict =  {
            "pixel_values": mtv_img,
            "path_img": mtv_path_img,
            "ctrl_img": mtv_condition,
        }

        if self.args.use_sdxl:
            data_dict.update(self.input_ids) 
        else:
            mtv_prompt = torch.cat(mtv_prompt, 0)
            data_dict.update({"input_ids": mtv_prompt}) 


        if self.trainorval == 'val':
            mtv_occ_rgb = torch.cat(mtv_occ_rgb, 0)
            data_dict["occ_rgb"] = mtv_occ_rgb
        elif self.trainorval == 'train':
            mtv_weight_mask = torch.cat(mtv_weight_mask, 0)
            data_dict["weight_mask"] = mtv_weight_mask

        return data_dict


    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.dataset[idx]
        if self.use_valid_flag:
            mask = info['valid_flag']
            gt_names = set(info['gt_names'][mask])
        else:
            gt_names = set(info['gt_names'])

        cat_ids = []
        fore_flag = 0
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
                fore_flag = 1
        if fore_flag == 0:
            # model background as two objects
            for _ in range (120):
                cat_ids.append(self.cat2id['background'])
        return cat_ids
    


    def _get_sample_indices(self):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations after class sampling.
        """
        class_sample_idxs = {cat_id: [] for cat_id in self.cat2id.values()}
        for idx in range(len(self.dataset)):
            sample_cat_ids = self.get_cat_ids(idx)
            for cat_id in sample_cat_ids:
                class_sample_idxs[cat_id].append(idx)
        duplicated_samples = sum(
            [len(v) for _, v in class_sample_idxs.items()])
        class_distribution = {
            k: len(v) / duplicated_samples
            for k, v in class_sample_idxs.items()
        }
        # print(class_sample_idxs, class_distribution)
        for key, value in class_sample_idxs.items():
            print(key, len(value))

        sample_indices = []

        frac = 1.0 / len(self.CLASSES)
        ratios = [frac / v for v in class_distribution.values()]
        for cls_inds, ratio in zip(list(class_sample_idxs.values()), ratios):
            sample_indices += np.random.choice(cls_inds,
                                               int(len(cls_inds) *
                                                   ratio)).tolist()
        return sample_indices
