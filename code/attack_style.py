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



class Region_Denoised_Classifier(torch.nn.Module):
    def __init__(self, diffusion, model, classifier, t, mask):
        super().__init__()
        self.diffusion = diffusion
        self.model = model
        self.classifier = classifier
        self.t = t
        self.mask = mask
    
    def sdedit(self, x, t, to_01=True, mask=None):

        # assume the input is 0-1
        
        x = x * 2 - 1
        
        t = torch.full((x.shape[0], ), t).long().to(x.device)
    
        x_t = self.diffusion.q_sample(x, t) 
        
        sample = x_t
    
        # print(x_t.min(), x_t.max())
    
        # si(x_t, 'vis/noised_x.png', to_01=True)
        
        indices = list(range(t+1))[::-1]
        
        # visualize 
        l_sample=[]
        l_predxstart=[]

        for i in indices:
            t_tensor = torch.full((x.shape[0], ), i).long().to(x.device)
            out = self.diffusion.ddim_sample(self.model, sample, t_tensor)
            sample = out["sample"]


            l_sample.append(out['sample'])
            l_predxstart.append(out['pred_xstart'])

            if i > 0:
                sample = sample * mask + self.diffusion.q_sample(x, t_tensor -1) * (1 - mask)
            else:
                sample = sample * mask + x * (1 - mask)
        
        
        # visualize
        si(torch.cat(l_sample), 'l_sample.png', to_01=1)
        si(torch.cat(l_predxstart), 'l_pxstart.png', to_01=1)

        # the output of diffusion model is [-1, 1], should be transformed to [0, 1]
        if to_01:
            sample = (sample + 1) / 2
        
        return sample
        
    
    def forward(self, x):
        
        assert self.mask is not None
        
        out = self.sdedit(x, self.t, True, self.mask) # [0, 1]
        out = self.classifier(out)
        return out
    
    


def generate_x_adv_denoised_region(x, y, diffusion, model, classifier, pgd_conf, device, t, mask):
    
    
    net = Region_Denoised_Classifier(diffusion, model, classifier, t, mask)

        
    net = wrapper(net, x * (1 - mask), mask)


    adversary = LinfPGDAttack(  net,
                                loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"), 
                                eps=pgd_conf['eps'],
                                nb_iter=pgd_conf['iter'], 
                                eps_iter=pgd_conf['alpha'], 
                                rand_init=False, 
                                targeted=False
                                
                                )
    
    # x_adv = adversary.perturb(x*mask, y*0+625)
    x_adv = adversary.perturb(x*mask, y)
    
    return x_adv + x * (1 - mask)






def style_transfer(x, x_refer, mask, content_w, style_w, num_iters=300):
    
    

    model, style_losses, content_losses = get_style_model_and_losses(x_refer, x, style_w, content_w)
    

    x = x.clone()
    input_param = nn.Parameter(x)

    # optimizer =  optim.SGD([input_param], lr=0.01, momentum=0.9)
    optimizer = optim.Adam([input_param], lr=0.01, )

    run = [0]

    while run[0] < num_iters:
        def closure():
            input_param.data.clamp_(0, 1)

            optimizer.zero_grad()
            input_param_new = input_param 
            model(input_param_new)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.backward()
            for cl in content_losses:
                content_score += cl.backward()

            run[0] += 1
            if run[0] % 10 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()
            
            return style_score + content_score
        
        optimizer.step(closure)
    
    
    input_param.data.clamp_(0, 1)

    return input_param




    
    
    


def Attack_Region_Style(exp_name, classifier, device, respace, t, eps=16, iter=10, name='attack_style_new'):
    
    
    pgd_conf = gen_pgd_confs(eps=eps, alpha=1, iter=iter, input_range=(0, 1))

    save_path = f'vis/{name}/{classifier}_eps{eps}_iter{iter}_{respace}_t{t}/'

    mp(save_path)

    save_path = f'vis/{name}/{classifier}_eps{eps}_iter{iter}_{respace}_t{t}/'

    mp(save_path)



    classifier = get_archs(classifier, 'imagenet')
    
    classifier = classifier.to(device)
    classifier.eval()
    

    

    model, diffusion = get_imagenet_dm_conf(device=device, respace=respace)
    
    
    c = 0
    a = 0



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
        

    
    # ========== lamp =========
    if exp_name == 'lamp':
        x = load_png(f'data/advcam_dataset/other_imgs/img/lamp.jpg', 224)[None, ...].to(device)
        mask = load_png(f'data/advcam_dataset/other_imgs/seg/lamp.jpg', 224)[None, ...].to(device)
        style_refer = load_png(f'data/advcam_dataset/other_imgs/img/lamp-style.jpg', 224)[None, ...].to(device)
    
    if exp_name == 'car2':
        x = load_png(f'data/advcam_dataset/other_imgs/img/car.jpg', 224)[None, ...].to(device)
        mask = load_png(f'data/advcam_dataset/other_imgs/seg/car-mask.jpg', 224)[None, ...].to(device)
        style_refer = load_png(f'data/advcam_dataset/other_imgs/img/redcar-style2.jpg', 224)[None, ...].to(device)
        
    
    if exp_name == 'car3':
        x = load_png(f'data/advcam_dataset/other_imgs/img/car.jpg', 224)[None, ...].to(device)
        mask = load_png(f'data/advcam_dataset/other_imgs/seg/car-mask.jpg', 224)[None, ...].to(device)
        style_refer = load_png(f'data/advcam_dataset/other_imgs/img/yellowcar-style.jpg', 224)[None, ...].to(device)
    
    
    if exp_name == 'ub':
        x = load_png(f'data/advcam_dataset/other_imgs/img/umbrella.jpg', 224)[None, ...].to(device)
        mask = load_png(f'data/advcam_dataset/other_imgs/seg/umbrella.jpg', 224)[None, ...].to(device)
        style_refer = load_png(f'data/advcam_dataset/other_imgs/img/umbrella-style.jpg', 224)[None, ...].to(device)
        
 
    if exp_name == 'dog':
        x = load_png(f'data/advcam_dataset/other_imgs/img/umbrella.jpg', 224)[None, ...].to(device)
        mask = load_png(f'data/advcam_dataset/other_imgs/seg/umbrella-mask.jpg', 224)[None, ...].to(device)
        style_refer = load_png(f'data/advcam_dataset/other_imgs/img/umbrella-style.jpg', 224)[None, ...].to(device)
        
    
    # customize your own style with a different exp_name
        


    
    mask = (mask > 0).float() # 1 means umasked, 0 means dont need to modify
    

    # do style transfer first
    x_s = style_transfer(x, style_refer, mask, content_w=1, style_w=4000, num_iters=1000)
    si(torch.cat([x, mask, style_refer, x_s*mask+x*(1-mask)], -1), save_path + f'/{exp_name}_style_trans.png')
 

    y_pred = classifier(x).argmax(1) # original prediction
    print(y_pred)
    x_s = x_s*mask+x*(1-mask)
    x_s = x_s.detach()

    # generate DIFF-PGD Samples
    x_adv_diff_region = generate_x_adv_denoised_region(x_s, y_pred, diffusion, model, classifier, pgd_conf, device, t, mask)
        
    # get purified sample
    with torch.no_grad():
        net = Region_Denoised_Classifier(diffusion, model, classifier, t, mask)
        x_adv_diff_p_region = net.sdedit(x_adv_diff_region, t, True, mask)

    print(classifier(x_adv_diff_region).argmax(1))
    y_final=classifier(x_adv_diff_p_region).argmax(1).item()
    
        
    si(torch.cat(
        [torch.cat([x, x_s, x_adv_diff_region, x_adv_diff_p_region, mask], -1),
            10*torch.cat([x-x, x_s-x, x_adv_diff_region-x_s, x_adv_diff_p_region-x_adv_diff_region, mask], -1)
        ],-2)
        , save_path + f'/{exp_name}_final{y_final}.png')


for exp_name in ['tra4', 'tra1', 'tra5', 'tra6', 'leaf', 'car', 'lamp', 'car']:
        
    Attack_Region_Style(exp_name, 'resnet50', 0, 'ddim40', t=4, eps=64, iter=10)

# for exp_name in ['car2']:
        
#     Attack_Region_Style(exp_name, 'resnet50', 0, 'ddim10', t=4, eps=96, iter=100)

