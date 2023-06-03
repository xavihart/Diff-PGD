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

from attack_tools import *


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
    
    x_adv = adversary.perturb(x*mask, y)
    
    return x_adv * mask + x * (1 - mask)






def Attack_Region(classifier, device, respace, t, eps=16, iter=10, name='attack_region', skip = 200, ratio=0.4):
    
    
    pgd_conf = gen_pgd_confs(eps=eps, alpha=1, iter=iter, input_range=(0, 1))

    save_path = f'vis/{name}/{classifier}_eps{eps}_iter{iter}_{respace}_t{t}/'

    mp(save_path)

    save_path = f'vis/{name}/{classifier}_eps{eps}_iter{iter}_{respace}_t{t}/'

    mp(save_path)



    classifier = get_archs(classifier, 'imagenet')
    
    classifier = classifier.to(device)
    classifier.eval()
    
    dataset = get_dataset(
        'imagenet', split='test'
    )
    

    model, diffusion = get_imagenet_dm_conf(device=device, respace=respace)
    
    
    c = 0
    a = 0

    for i in range(dataset.__len__()):
        if i % skip != 0:
            continue


        
        time_st = time.time()
        print(f'{c}/{dataset.__len__()//skip}')


        x, y = dataset[i]
        x = x[None, ].to(device)
        y = torch.tensor(y)[None, ].to(device)
        
        y_pred = classifier(x).argmax(1) # original prediction
                    
        # pgd attack_region

        region_mask = 1 - gen_mask(x, type='square', ratio=ratio) # same shape as x, remain a square valued 1, other 0


        x_pgd_region = gen_region_pgd_sample(classifier, x, region_mask, iter=iter, eps=eps)

        x_pgd_region_pred = classifier(x_pgd_region).argmax(1)

        if x_pgd_region_pred != y_pred:
            a += 1
        
        c += 1

  
        # Generate Diff-rPGD samples: x^n
        x_adv_diff_region = generate_x_adv_denoised_region(x, y_pred, diffusion, model, classifier, pgd_conf, device, t, region_mask)
        
        # x^n -> x^n_0
        with torch.no_grad():
            net = Region_Denoised_Classifier(diffusion, model, classifier, t, region_mask)
            x_adv_diff_p_region = net.sdedit(x_adv_diff_region, t, True, region_mask)

        
        
        si(torch.cat(
            [torch.cat([x_pgd_region, x_adv_diff_region, x_adv_diff_p_region, region_mask], -1),
             100*torch.cat([x-x_pgd_region, x-x_adv_diff_region, x-x_adv_diff_p_region, region_mask], -1)
            ],-2)
            , save_path + f'/{i}.png')
        

        pkg = {
            'x': x,
            'y': y,
            'x_adv': x_adv_diff_region,
            'x_adv_diff': x_adv_diff_p_region,
            'x_pgd': x_pgd_region
        }


        torch.save(pkg, save_path + f'{i}.bin')
        

        print(classifier(x_adv_diff_region).argmax(1)==y_pred,
              classifier(x_adv_diff_p_region).argmax(1)==y_pred,
              classifier(x_pgd_region).argmax(1)==y_pred)
    
        print((x_pgd_region - x).abs().max(), 
              (x_adv_diff_region - x).abs().max(),
              (x_adv_diff_p_region - x).abs().max())




        # our repain attack

    print(a/c)

        



        


# Attack_Region('resnet50', 2, 'ddim30', t=2, eps=32, iter=20)
Attack_Region('resnet50', 4, 'ddim100', t=2, eps=32, iter=20)
