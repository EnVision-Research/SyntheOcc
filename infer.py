import torch

import cv2
import os
import tqdm
import pickle
import numpy as np
from torchvision.io import read_image

import torchvision.transforms.functional as tf

import torch.distributed as dist
import torch.multiprocessing as mp

from models.controlnet1x1 import ControlNetModel1x1 as ControlNetModel
from models.pipeline_controlnet_1x1_4dunet import (
    StableDiffusionControlNetPipeline1x1 as StableDiffusionControlNetPipeline,
)

from models.unet_2d_condition_multiview import UNet2DConditionModelMultiview
from diffusers import UniPCMultistepScheduler

from args_file import parse_args
from transformers import AutoTokenizer

from utils.dataset_nusmtv import NuScenesDatasetMtvSpar as NuScenesDataset


args = parse_args()

base_model_path = "/hpc2hdd/home/lli181/long_video/animate-anything/download/AI-ModelScope/stable-diffusion-v2-1"


ckp_path = "./exp/out_sd21_cbgs_loss/"


if args.model_path_infer is not None:
    ckp_path = args.model_path_infer

if "checkpoint" not in ckp_path:
    dirs = os.listdir(ckp_path)
    dirs = [d for d in dirs if d.startswith("checkpoint")]
    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    ckp_path = os.path.join(ckp_path, dirs[-1]) if len(dirs) > 0 else ckp_path

height = 448
width = 800


generator = torch.manual_seed(0)
# generator = torch.manual_seed(666)

num_validation_images = 2


tokenizer = AutoTokenizer.from_pretrained(
    args.pretrained_model_name_or_path,
    subfolder="tokenizer",
    revision=args.revision,
    use_fast=False,
)

val_dataset = NuScenesDataset(args, tokenizer, args.gen_train_or_val)

CAM_NAMES = val_dataset.CAM_NAMES


save_path = "vis_dir/out_sd21_cbgs_loss2_40/samples"


if args.model_path_infer is not None:
    save_path = args.save_img_path


print(ckp_path, save_path)


for cam_name in CAM_NAMES:
    os.makedirs(os.path.join(save_path, cam_name), exist_ok=True)


def run_inference(rank, world_size, pred_results, input_datas, pipe, args):

    pipe.to("cuda")

    all_list = input_datas[rank]

    validation_prompts = []
    validation_prompts.append("show a photorealistic street view image.")

    with torch.no_grad():
        for img_idx in tqdm.tqdm(all_list):

            data_dict = val_dataset.__getitem__(img_idx)
            mtv_condition = data_dict["ctrl_img"].to("cuda")

            cfg_scale = torch.tensor([7.5]).to("cuda")

            images_tensor = []

            for iter_idx in range(len(validation_prompts)):

                curr_prompt = [validation_prompts[iter_idx]] * 6

                with torch.autocast("cuda"):
                    image = pipe(
                        prompt=curr_prompt,
                        image=mtv_condition,
                        num_inference_steps=20,
                        generator=generator,
                        height=height,
                        width=width,
                        controlnet_conditioning_scale=1.0,
                        guidance_scale=cfg_scale,
                    ).images  # [0]

                for mtv_idx, img in enumerate(image):
                    img = img.resize((1600, 900))

                    img_name = data_dict["path_img"][mtv_idx].split("/")[-1]
                    img.save(f"{save_path}/{CAM_NAMES[mtv_idx]}/{img_name}")

                image = torch.cat([torch.tensor(np.array(ii)) for ii in image], 1)

                images_tensor.append(image)

            # [448, 6, 800, 3] to [448, 4800, 3]
            raw_img = (
                data_dict["pixel_values"]
                .permute(2, 0, 3, 1)
                .reshape(images_tensor[0].shape)
                * 255
            )
            occ_rgb = (
                data_dict["occ_rgb"].permute(1, 0, 2, 3).reshape(images_tensor[0].shape)
            )
            gen_img = torch.cat(images_tensor, 0)
            gen_img = torch.cat([occ_rgb, gen_img, raw_img], 0)

            out_path = os.path.join(
                f"{save_path}/{img_idx:06d}_{str(cfg_scale.item())}.jpg"
            )

            cv2.imwrite(
                out_path, cv2.cvtColor(gen_img.cpu().numpy(), cv2.COLOR_RGB2BGR)
            )


if __name__ == "__main__":
    os.system("export NCCL_SOCKET_IFNAME=eth1")

    from torch.multiprocessing import Manager

    world_size = 4
    # world_size = 8

    all_len = len(val_dataset)
    # all_len = 500

    all_list = np.arange(0, all_len, 1)

    all_len_sel = all_list.shape[0]
    val_len = all_len_sel // world_size * world_size

    all_list_filter = all_list[:val_len]

    all_list_filter = np.split(all_list_filter, world_size)

    input_datas = {}
    for i in range(world_size):
        input_datas[i] = list(all_list_filter[i])
        print(len(input_datas[i]))

    input_datas[0] += list(all_list[val_len:])

    controlnet = ControlNetModel.from_pretrained(
        ckp_path, subfolder="controlnet", torch_dtype=torch.float16
    )
    unet = UNet2DConditionModelMultiview.from_pretrained(
        ckp_path, subfolder="unet", torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_path, unet=unet, controlnet=controlnet, torch_dtype=torch.float16
    )

    # speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.enable_xformers_memory_efficient_attention()
    pipe.set_progress_bar_config(disable=True)
    from diffusers.models.attention_processor import AttnProcessor2_0

    pipe.unet.set_attn_processor(AttnProcessor2_0())

    run_inference(args.curr_gpu, 1, None, input_datas, pipe, args)

    # with Manager() as manager:
    #     pred_results = manager.list()
    #     mp.spawn(run_inference, nprocs=world_size, args=(world_size,pred_results,input_datas,pipe,args,), join=True)
