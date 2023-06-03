from load_dm import get_imagenet_dm_conf
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
import glob
import torch.nn.functional as F
# import wandb

# wandb.init(
#     project='diff-pgd',
#     name='attack_physics'
# )


def physics_adapter(patch, patch_mask, bkg_dir, scale_range=(0.3, 0.4), margin=0.4):   
    assert len(patch.shape) == 4
    
    # randomly select one bkg
    
    bkg_all = glob.glob(bkg_dir+'/*.jpg')
    bkg_all += glob.glob(bkg_dir+'/*.png')

    bkg_selected = random.choice(bkg_all)
    bkg = load_png(bkg_selected, 224)[None, ...].to(patch.device) # 1 * 3 * h * w
    h_bkg, w_bkg = bkg.shape[-2], bkg.shape[-1]
    
    # scale and transfer, no rotation
    scale_min, scale_max = scale_range
    t = random.randrange(0, 100)/100
    scale = scale_min + t * (scale_max - scale_min)
    
    h, w = int(patch.shape[-2]*scale), int(patch.shape[-1]*scale)
    
    patch = F.interpolate(patch, (h, w), mode='bilinear')
    mask  = F.interpolate(patch_mask, (h, w), mode='bilinear')
    
    while True:
        x_idx, y_idx = random.uniform(margin, 1-margin), random.uniform(margin, 1-margin)
        x_idx = int(x_idx * h_bkg)
        y_idx = int(y_idx * w_bkg)
        if x_idx + h < h_bkg and y_idx + w < w_bkg:
            break
    
    x = bkg.clone()
    x[:, :, x_idx:x_idx+h, y_idx:y_idx+w] = x[:, :, x_idx:x_idx+h, y_idx:y_idx+w] * (1 - mask) +  mask * patch
    
    # light change
    color_trans = transforms.ColorJitter(brightness=0.2)
    x = color_trans(x)
    
    si(x, 'adv_phys.png')
    
    return x



        


class Region_Denoised_Classifier(torch.nn.Module):
    def __init__(self, diffusion, model, classifier, t, mask):
        super().__init__()
        self.diffusion = diffusion
        self.model = model
        self.classifier = classifier
        self.t = t
        self.mask = mask
    
    
    def sdedit(self, x_, t, to_01=True, mask=None):
        

        # assume the input is 0-1
        # return x
        x = x_ * 2 - 1
        
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

        # assert self.mask is not None
        
        out = self.sdedit(x, self.t, True, self.mask) # [0, 1]
        # out = self.classifier(out)
        
        return out
    


# def test():
#     pass

# def train():



def Attack_Physics(mode, bkg_name, patch_name, classifier, device, respace, t, target, c_w, s_m, w_adv=1, iter=10, name='attack_physics'):

    assert mode in ['diff-pgd', 'adv-patch', 'advcam']
    
    
    # pgd_conf = gen_pgd_confs(eps=eps, alpha=1, iter=iter, input_range=(0, 1))

    save_path = f'vis/{name}_{mode}/{bkg_name}-{patch_name}/{classifier}_iter{iter}_{respace}_t{t}/'

    mp(save_path)


    classifier = get_archs(classifier, 'imagenet')
    
    classifier = classifier.to(device)
    classifier.eval()
    
    # dataset = get_dataset(
    #     'imagenet', split='test'
    # )
    
    if mode == 'diff-pgd':
        model, diffusion = get_imagenet_dm_conf(device=device, respace=respace)
    
    if bkg_name == 't-shirt':
        bkg_dir = f'data/our_dataset/{bkg_name}/'
        y_pred = 610 # jersey, t-shirt
        scale_range=(0.4, 0.5)
        margin=0.1
    elif bkg_name == 'bag':
        bkg_dir = f'data/our_dataset/{bkg_name}/'
        y_pred = 414 # bag
        scale_range=(0.2, 0.3)
        margin=0.3
    elif bkg_name == 'keyboard':
        bkg_dir = f'data/our_dataset/{bkg_name}/'
        y_pred = 508 # keyboard
        scale_range=(0.2, 0.3)
        margin=0.2
    elif bkg_name == 'computer_mouse':
        bkg_dir = f'data/our_dataset/{bkg_name}/'
        y_pred = 673 # mouse
        scale_range=(0.2, 0.3)
        margin=0.2
    else:
        raise "Unavailable Background!"
    
    # add your own dataset here


        
    # load patch and mask
    patch = load_png(f'data/our_dataset/patches/{patch_name}.png', 224)[None, ].to(device)
    if os.path.exists(f'data/our_dataset/patches/{patch_name}_mask.png'):
        patch_mask = load_png(f'data/our_dataset/patches/{patch_name}_mask.png', 224)[None, ].to(device)
    else:
        patch_mask = torch.ones(1, 3, 224, 224).to(device)


    x = patch.clone()
    d = torch.zeros_like(x).to(device)
    d.requires_grad = True
    
    if target is None:
        y_target = None
    else:
        y_target = target

    if mode == 'diff-pgd':
        edit_net = Region_Denoised_Classifier(diffusion, model, None, t, patch_mask)


    model, style_losses, content_losses = get_style_model_and_losses(x, x, 0, c_w)

    

    optimizer = optim.Adam([d], lr=0.01, )
    loss_l = []
    
    for iter_id in tqdm(range(iter)):
        optimizer.zero_grad()

        if mode == 'diff-pgd':
            x_new = edit_net(x + d*patch_mask)
        else:
            x_new = x + d * patch_mask
        
        x_new = torch.clip(x_new, 0, 1)

        
        phy_x = physics_adapter(x_new, patch_mask, bkg_dir, scale_range=scale_range, margin=margin)
        out = classifier(phy_x)
        loss = out[:, y_pred]

        if y_target is not None:
            loss -= out[:, y_target]
        

        
        loss.backward()


        loss_l.append(loss.item())

        

        optimizer.step()
        plt.plot(loss_l)
        plt.savefig(save_path + f'{patch_name}-{bkg_name}_loss.png')
        plt.close()
        

        
        if mode == 'diff-pgd':
            x_p = edit_net(x + d)
        else:
            x_p = x + 1
        # x_p = x        
        if iter_id % 100 == 0:

            si(torch.cat([x, torch.clip(x+d, 0, 1), x_p], -1), save_path + f'{patch_name}-{bkg_name}-{iter_id}.png')
        




######################## adv-patch #####################################
# Attack_Physics(mode='adv-patch', bkg_name='computer_mouse', patch_name='apple', classifier='resnet50', \
#                 device=0, respace=None, t=None, target=None, c_w=0, s_m=0, iter=4000, name='attack_physics')

# Attack_Physics(mode='adv-patch', bkg_name='bag', patch_name='cat', classifier='resnet50', \
#                 device=0, respace=None, t=None, target=187, c_w=0, s_m=0, iter=4000, name='attack_physics')
######################## adv-patch #####################################



######################## advcam #####################################
# Attack_Physics(mode='advcam', bkg_name='computer_mouse', patch_name='apple', classifier='resnet50', \
#                 device=0, respace=None, t=None, target=None, c_w=1, s_m=0.01, iter=4000, name='attack_physics')

# Attack_Physics(mode='advcam', bkg_name='bag', patch_name='cat', classifier='resnet50', \
#                 device=0, respace=None, t=None, target=187, c_w=1, s_m=0.01, iter=4000, name='attack_physics')
# ######################## advcam #####################################




####################### diff-pgd #####################################
Attack_Physics(mode='diff-pgd', bkg_name='computer_mouse', patch_name='apple', classifier='resnet50', \
                device=0, respace='ddim10', t=2, target=None, c_w=1, s_m=0.01, iter=4000, name='attack_physics')
####################### diff-pgd #####################################

