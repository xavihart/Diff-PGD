from load_dm import get_imagenet_dm_conf
from dataset import get_dataset
from utils import *
import torch
import torchvision
from tqdm.auto import tqdm
import random
from archs import get_archs, IMAGENET_MODEL
from advertorch.attacks import LinfPGDAttack
import matplotlib.pylab as plt
import time
import glob
import torchvision.models as models


from attack_tools import *
import torchvision.transforms as transforms
from PIL import Image

import torch.optim as optim

import argparse





def adv_patch():
    
    pass

def smooth_loss(output, weight):
    tv_loss = torch.sum(
        (output[:, :-1, :-1, :] - output[:, :-1, 1:, :]) * (output[:, :-1, :-1, :] - output[:, :-1, 1:, :]) + \
        (output[:, :-1, :-1, :] - output[:, 1:, :-1, :]) * (output[:, :-1, :-1, :] - output[:, 1:, :-1, :])) / 2.0
    return tv_loss * weight

    
    
def adv_cam(classifier, bkg, x_raw, mask, style=False, ref=None, iters=1000, adv_weight=0, smooth_weight=1, style_weight=4000, content_weight=1):
    
    
    # original settings: adv_weight:1000-5000, smooth_weight=1e-3, style_weight=100, content_weight=5
    y = classifier(x_raw).argmax(1)
    
    x_raw_copy = x_raw.clone()
    
    
    x_raw.requires_grad = True
    
    
    
    optimizer = optim.Adam([x_raw], lr=0.01, )
    
    model, style_losses, content_losses = get_style_model_and_losses(ref, x_raw, style_weight, content_weight)
    
    for iter_id in range(iters):
        
        
        
        print(iter_id)
        
        optimizer.zero_grad()
        
        x = x_raw * mask + x_raw_copy *  (1 - mask)
        
        x.data.clamp_(0, 1)
        
        loss = 0.
            
        
        if style == True:
            
            model(x)
            style_score = 0
            content_score = 0
            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss
            loss += (content_score + style_score)
            
        loss += smooth_loss(x, smooth_weight)
        
        loss += classifier(x)[:, y.item()][0] * adv_weight
        
        loss.backward()
        
        optimizer.step()
        
    x_raw.data.clamp_(0, 1)
        
    return x_raw
        

def main(exp_name, classifier, device, iters, w_a, w_s, w_c, w_sm, name='advcam'):

    save_path = f'vis/{name}/{classifier}_wa{w_a}_wst{w_s}_wc{w_c}_w_sm{w_sm}_iter{iters}/'

    mp(save_path)


    classifier = get_archs(classifier, 'imagenet')
    
    classifier = classifier.to(device)
    classifier.eval()
    
    


    if 'tra' in exp_name:
        x = load_png(f'data/advcam_dataset/other_imgs/img/{exp_name}.jpg', 224)[None, ...].to(device)
        mask = load_png(f'data/advcam_dataset/other_imgs/seg/{exp_name}.jpg', 224)[None, ...].to(device).expand(1, 3, 224, 224)
        style_refer = load_png(f'data/advcam_dataset/other_imgs/img/{exp_name}-style.jpg', 224)[None, ...].to(device)
        
    # =========== leave ========

    if exp_name == 'leaf':
        x = load_png(f'data/advcam_dataset/other_imgs/img/leaf.jpg', 224)[None, ...].to(device)
        mask = load_png(f'data/advcam_dataset/other_imgs/seg/leaf.jpg', 224)[None, ...].to(device)
        style_refer = load_png(f'data/advcam_dataset/other_imgs/img/leaf-style.jpg', 224)[None, ...].to(device)
        


    # =========== car   ========
    if exp_name == 'car':
        x = load_png(f'data/advcam_dataset/other_imgs/img/car.jpg', 224)[None, ...].to(device)
        mask = load_png(f'data/advcam_dataset/other_imgs/seg/car-mask.jpg', 224)[None, ...].to(device)
        style_refer = load_png(f'data/advcam_dataset/other_imgs/img/car-style.jpg', 224)[None, ...].to(device)
    
    if exp_name == 'car2':
        x = load_png(f'data/advcam_dataset/other_imgs/img/car.jpg', 224)[None, ...].to(device)
        mask = load_png(f'data/advcam_dataset/other_imgs/seg/car-mask.jpg', 224)[None, ...].to(device)
        style_refer = load_png(f'data/advcam_dataset/other_imgs/img/redcar-style2.jpg', 224)[None, ...].to(device)
        
    
    if exp_name == 'car3':
        x = load_png(f'data/advcam_dataset/other_imgs/img/car.jpg', 224)[None, ...].to(device)
        mask = load_png(f'data/advcam_dataset/other_imgs/seg/car-mask.jpg', 224)[None, ...].to(device)
        style_refer = load_png(f'data/advcam_dataset/other_imgs/img/yellowcar-style.jpg', 224)[None, ...].to(device)
        

    # ========== lamp =========
    if exp_name == 'lamp':
        x = load_png(f'data/advcam_dataset/other_imgs/img/lamp.jpg', 224)[None, ...].to(device)
        mask = load_png(f'data/advcam_dataset/other_imgs/seg/lamp.jpg', 224)[None, ...].to(device)
        style_refer = load_png(f'data/advcam_dataset/other_imgs/img/lamp-style.jpg', 224)[None, ...].to(device)
        

    x_adv_cam = adv_cam(classifier, None, x.clone(), mask, style=True, ref=style_refer,
                        adv_weight=w_a, style_weight=w_s, content_weight=w_c, smooth_weight=w_sm, iters=iters)

    si(torch.cat([x, style_refer, x_adv_cam], -1), save_path + f'{exp_name}.png')
    
    print(classifier(x).argmax(1))
    
    print(classifier(x_adv_cam).argmax(1))
    
    
# main('resnet50', 0, iters=1000, w_a=1000, w_s=1e-3, w_c=5, w_sm=1e-3)
# for exp_name in ['tra4', 'tra1', 'tra5', 'tra6', 'leaf', 'car', 'lamp']:
#     main(exp_name, 'resnet50', 0, iters=500, w_a=0, w_s=4000, w_c=1, w_sm=0)


for exp_name in ['car2', 'car3']:
    main(exp_name, 'resnet50', 0, iters=500, w_a=0, w_s=4000, w_c=1, w_sm=0)
