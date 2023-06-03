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



class Denoiser(nn.Module):
    def __init__(self, 
                 dataset, 
                 sigma, 
                 sigma_scale=True,
                 rescale=False,
                 rescale_step=10,
                 classifier=None,
                 sigma_must_include=None,
                 guided=False,
                 guided_scale=None,
                 return_type=None,
                 ) -> None:
        super().__init__()

        self.sigma_scale=sigma_scale
        self.dataset = dataset
        
        self.rescale = rescale
        self.rescale_step = rescale_step
        self.classifier = classifier

        self.sigma_must_include = sigma_must_include

        
        # conditianl 
        self.class_cond = guided
        self.classifier_scale = guided_scale

        # voting
        self.return_type = return_type


        self.timestep_respacing = [self.rescale_step] if self.rescale else None
        

        if dataset == 'imagenet': # using guided diffusion repo

            from guided_diffusion.script_util import (
                NUM_CLASSES,
                model_and_diffusion_defaults,
                classifier_defaults,
                create_model_and_diffusion,
                create_classifier,
                add_dict_to_argparser,
                args_to_dict,
            )

            def create_argparser():
                defaults = dict(
                    clip_denoised=True,
                    num_samples=10000,
                    batch_size=16,
                    use_ddim=False,
                    model_path="",
                    classifier_path="",
                    classifier_scale=1.0,
                    sigma=sigma,
                    skip=100
                )

                model_config = dict(
                    use_fp16=False,
                    attention_resolutions="32, 16, 8",
                    class_cond=False if not self.class_cond  else True,
                    diffusion_steps=1000,
                    image_size=256,
                    learn_sigma=True,
                    noise_schedule='linear',
                    num_channels=256,
                    num_head_channels=64,
                    num_res_blocks=2,
                    resblock_updown=True,
                    use_scale_shift_norm=True,
                    timestep_respacing=self.timestep_respacing,
                    sigma_must_include=self.sigma_must_include,
                    classifier_scale=self.classifier_scale
                )

                defaults.update(model_and_diffusion_defaults())
                defaults.update(model_config)
                # defaults.update(classifier_defaults())
                args = Namespace(**defaults)
                # print(args)
                return args
            
            self.args = create_argparser()

            self.model, self.diffusion = create_model_and_diffusion(
            **args_to_dict(self.args, model_and_diffusion_defaults().keys())
             )
            
        elif 'cifar10' in dataset: # using improved diffusion repo

            from improved_diffusion.script_util import (
                NUM_CLASSES,
                model_and_diffusion_defaults,
                create_model_and_diffusion,
                add_dict_to_argparser,
                args_to_dict,
            )

            def create_argparser():
                defaults = dict(
                    clip_denoised=True,
                    num_samples=10000,
                    batch_size=16,
                    use_ddim=False,
                    model_path="/home/hxue45/data/sem1/Diff-Smoothing/model/dm/cifar10_uncond_50M_500K.pt",
                    classifier_path="",
                    classifier_scale=1.0,
                    sigma=sigma,
                    skip=100
                )

                model_config = dict(
                    use_fp16=False,
                    image_size=32,
                    learn_sigma=True,
                    num_channels=128,
                    num_res_blocks=3,
                    diffusion_steps=4000,
                    noise_schedule='cosine',
                    timestep_respacing= self.timestep_respacing,
                    sigma_must_include=self.sigma_must_include
                )
    
                defaults.update(model_and_diffusion_defaults())
                # defaults.update(classifier_defaults())
                defaults.update(model_config)
                # defaults.update(classifier_defaults())
                args = Namespace(**defaults)
                return args
            
            self.args = create_argparser()
            self.model, self.diffusion = create_model_and_diffusion(
            **args_to_dict(self.args, model_and_diffusion_defaults().keys())
             )


        else:
            raise Exception("Dataset must be in ['imagenet', 'cifar10']")
        
    @th.no_grad()
    def forward(self, x, y=None, sigma=None, one_step=False):
        # here x is the normalized clean image ranged from -1, 1
        # we should add noise to this image first, remaining is the same
        x_raw = x

        print(sigma)
        
        x = x + th.randn_like(x, device=x.device) * sigma * 2
        

        time_st = time.time()

        if sigma is not None:
            sigma = sigma
        else:
            sigma = self.args.sigma

        if self.sigma_scale:
            sigma *= 2
        
        
        b = x.shape[0]

        x_noised = x
        x_noised_raw = x_noised
        t = np.abs(self.diffusion.alphas_cumprod - 1 / (1 + sigma ** 2)).argmin()


        x_noised = x_noised * np.sqrt(self.diffusion.alphas_cumprod[t])
        t_b = th.full((b, ), t).long()
        x_noised = x_noised.float()


        # model = model.cuda()
        x_noised = x_noised.to(x.device)
        t_b = t_b.to(x.device)
        # print(x_noised.dtype, t_b.dtype)
        
        time_sp = time.time()
        
        model_kwargs={}
        cond_fn_ = None
        
        # prepare gradient calculator
        if self.class_cond:
            def cond_fn(x, t, y=None):
                assert y is not None
                with th.enable_grad():
                    x_in = x.detach().requires_grad_(True)
                    logits = self.classifier(x_in)
                    log_probs = F.log_softmax(logits, dim=-1)
                    selected = log_probs[range(len(logits)), y.view(-1)]
                    return th.autograd.grad(selected.sum(), x_in)[0] * self.classifier_scale
            if y is not None:
                y = y.to(x.device)
            model_kwargs['y'] = y
            cond_fn_ = cond_fn
        
        x_denoised = None
        
        if self.ddim:
            sample_func = self.diffusion.ddim_sample
        else:
            sample_func = self.diffusion.p_sample
        
        if not self.rescale:
            x_denoised = sample_func(self.model, 
                                                 x_noised, 
                                                 t_b)['pred_xstart']
            
        else:

            x_sample = x_noised 
            t_sample = t_b
            x_denoised = None if self.return_type == 'single' else []

            if one_step:
                t = 0

            for _ in range(t+1):
                if 'cifar10' in self.dataset:
                    out = sample_func(self.model, x_sample, t_sample)
                else:
                    out = sample_func(self.model, x_sample, t_sample, cond_fn=cond_fn_,
                                                    model_kwargs=model_kwargs)
                
                x_sample = out['sample']
                # th.save(x_sample, 'x_sample_dp.bin')
                # exit(0)

                if self.return_type == 'single' or one_step:
                    x_denoised = out['pred_xstart']
                elif self.return_type == 'pool':
                    x_denoised.append(out['pred_xstart'])
                else:
                    raise TypeError('Return type must be in single / pool')

                t_sample = t_sample - 1 # note must be long
            
            if self.return_type == 'pool' and (not one_step):
                x_denoised = th.cat(x_denoised, 0) # return pool to vote
            
        
        
        image_show = 0.5 * (th.cat([x[0].cpu(), x_noised_raw[0].cpu(), x_denoised[0].cpu(), x[0].cpu()-x_denoised[0].cpu()], 1) + 1)



        if 'vit' in self.dataset and 'cifar' in self.dataset:
            # resize to 224 * 224
            resize = torchvision.transforms.Resize(224)
            x_denoised = resize(0.5 * (x_denoised + 1))
        
        else:
            x_denoised = 0.5 * (x_denoised + 1)

        return x_denoised


def get_imagenet_dm_conf(class_cond=False, respace=""):

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
    args = create_argparser()

    model, diffusion = create_model_and_diffusion(
    **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    return model, diffusion




