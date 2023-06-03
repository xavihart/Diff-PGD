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
from pytorch_clip_guided_loss import get_clip_guided_loss
    


def Attack_Region(classifier, device, respace, t, eps=16, iter=10, name='attack_style'):
    
    
    pgd_conf = gen_pgd_confs(eps=eps, alpha=1, iter=iter, input_range=(0, 1))

    save_path = f'vis/{name}/{classifier}_eps{eps}_iter{iter}_{respace}_t{t}/'

    mp(save_path)

    save_path = f'vis/{name}/{classifier}_eps{eps}_iter{iter}_{respace}_t{t}/'

    mp(save_path)



    classifier = get_archs(classifier, 'imagenet')
    
    classifier = classifier.to(device)
    classifier.eval()
    
    # dataset = get_dataset(
    #     'imagenet', split='test'
    # )
    

    model, diffusion = get_imagenet_dm_conf(device=device, respace=respace)
    
    skip = 200
    c = 0
    a = 0



    x = load_png(p='physical-attack-data/content/off-target/off-target5.jpg', size=224)[None, ...].to(device)
    mask = load_png(p='physical-attack-data/content-mask/off-target5.jpg', size=224)[None, ...].to(device) # red flower
    style_refer = load_png(p='physical-attack-data/style/off-target/red_flower.jpg', size=224)[None, ...].to(device)
    
    
    
    # text-based style transfer
    loss_fn = get_clip_guided_loss(clip_type="ruclip", input_range = (-1, 1)).eval().requires_grad_(False)
    loss_fn.add_prompt(text="blue flowers")
    loss_fn.to(device)
    # loss_fn.add_prompt(image=x)
    
    x_t_g = x.clone()
    
    x_t_g.requires_grad = True
    
    optimizer = optim.SGD([x_t_g], lr=0.1, momentum=0.99)
    iters=500
    
    
    for _ in range(iters):
        optimizer.zero_grad()
        image_ = x_t_g * mask
        loss = loss_fn.image_loss(image=image_)["loss"]
        print(loss)
        loss.backward()
        optimizer.step()
    
    si(torch.cat([x, x_t_g, x-x_t_g]), 'text_guided.png')
    exit(0)
        
    
    
    



    # x = load_png(p='physical-attack-data/content/backpack/backpack3.jpg', size=224)[None, ...].to(device)
    # mask = load_png(p='physical-attack-data/content-mask/backpack3.jpg', size=224)[None, ...].to(device)
    # style_refer = load_png(p='physical-attack-data/style/backpack/15.jpg', size=224)[None, ...].to(device) # blue

    # style_refer = load_png('physical-attack-data/style/t-shirt/tar16.png', size=224)[None, ...].to(device) # fire style



    
    mask = (mask > 0).float() # 1 means umasked, 0 means dont need to modify

    x_s = style_transfer(x, style_refer, mask, content_w=10, style_w=2000, num_iters=300)
    si(torch.cat([x, mask, style_refer, x_s*mask+x*(1-mask)], -1), 'style_trans.png')
    # style_sdedit = Style_SDEdit(diffusion, model, None, x_ref=style_refer, x_content=x)
    # x_p = style_sdedit.style_sdedit(t, mask)

    # si(torch.cat([x, style_refer, mask, x_p, 100*(x_p-x)], -1), 'x_style_trans.png')
    # exit(0)
    


    y_pred = classifier(x).argmax(1) # original prediction
    print(y_pred)
    x_s = x_s*mask+x*(1-mask)
    x_s = x_s.detach()
    x_adv_diff_region = generate_x_adv_denoised_region(x_s, y_pred, diffusion, model, classifier, pgd_conf, device, t, mask)
        

    net = Region_Denoised_Classifier(diffusion, model, classifier, t, mask)
    x_adv_diff_p_region = net.sdedit(x_adv_diff_region, t, True, mask)

    print(classifier(x_adv_diff_region).argmax(1))
    print(classifier(x_adv_diff_p_region).argmax(1))
    
        
    si(torch.cat(
        [torch.cat([x, x_s, x_adv_diff_region, x_adv_diff_p_region, mask], -1),
            10*torch.cat([x-x, x-x_s, x_s-x_adv_diff_region, x_s-x_adv_diff_p_region, mask], -1)
        ],-2)
        , save_path + f'/backpack.png')
    
    
    # si(torch.cat(
    #     [torch.cat([x_pgd_region, x_adv_diff_region, x_adv_diff_p_region, region_mask], -1),
    #         torch.cat([x-x_pgd_region, x-x_adv_diff_region, x-x_adv_diff_p_region, region_mask], -1)
    #     ],-2)
    #     , save_path + f'/{i}.png')
    

    # pkg = {
    #     'x': x,
    #     'y': y,
    #     'x_adv': x_adv_diff_region,
    #     'x_adv_diff': x_adv_diff_p_region,
    #     'x_pgd': x_pgd_region
    # }


    
    # torch.save(pkg, save_path+f'{i}.bin')
    

    # print(classifier(x_adv_diff_region).argmax(1)==y_pred,
    #       classifier(x_adv_diff_p_region).argmax(1)==y_pred,
    #       classifier(x_pgd_region).argmax(1)==y_pred)
    # print((x_pgd_region - x).abs().max(), 
    #       (x_adv_diff_region - x).abs().max(),
    #       (x_adv_diff_p_region - x).abs().max())




    # our repain attack

# print(a/c)

    



        


        
Attack_Region('resnet50', 0, 'ddim30', t=4, eps=64, iter=100)

