from guided_diffusion.script_util import (
                NUM_CLASSES,
                model_and_diffusion_defaults,
                classifier_defaults,
                create_model_and_diffusion,
                create_classifier,
                add_dict_to_argparser,
                args_to_dict,
            )

import torch as th
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
import torch.nn as nn
from torchvision import transforms as T, utils
import numpy as np
import argparse
import os
from argparse import Namespace
import time
import datetime
from utils import *




def get_imagenet_dm_conf(class_cond=False, respace="", device='cuda',
                         model_path='/ssdscratch/hxue45/data/Diff-Smoothing/ckpt/dm/256x256_diffusion_uncond.pt'):

    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
    )

    model_config = dict(
            use_fp16=False,
            attention_resolutions="32, 16, 8",
            class_cond=class_cond,
            diffusion_steps=1000,
            image_size=256,
            learn_sigma=True,
            noise_schedule='linear',
            num_channels=256,
            num_head_channels=64,
            num_res_blocks=2,
            resblock_updown=True,
            use_scale_shift_norm=True,
            timestep_respacing=respace,
        )

    defaults.update(model_and_diffusion_defaults())
    defaults.update(model_config)
    args = Namespace(**defaults)
    

    model, diffusion = create_model_and_diffusion(
    **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    cprint('Create DM         ---------------', 'y')
    
    
    # load ckpt
    
    ckpt = th.load(model_path)
    model.load_state_dict(ckpt)
    model = model.to(device)
    
    
    cprint('Load DM Ckpt      ---------------', 'y')
    
    return model, diffusion





