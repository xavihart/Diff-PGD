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

from attack_tools import gen_pgd_confs



class Denoised_Classifier(torch.nn.Module):
    def __init__(self, diffusion, model, classifier, t):
        super().__init__()
        self.diffusion = diffusion
        self.model = model
        self.classifier = classifier
        self.t = t
    
    def sdedit(self, x, t, to_01=True):

        # assume the input is 0-1
        t_int = t
        
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

            # out = self.diffusion.ddim_sample(self.model, sample, t)           
            out = self.diffusion.ddim_sample(self.model, sample, torch.full((x.shape[0], ), i).long().to(x.device))


            sample = out["sample"]


            l_sample.append(out['sample'])
            l_predxstart.append(out['pred_xstart'])
        
        
        # visualize
        si(torch.cat(l_sample), 'l_sample.png', to_01=1)
        si(torch.cat(l_predxstart), 'l_pxstart.png', to_01=1)

        # the output of diffusion model is [-1, 1], should be transformed to [0, 1]
        if to_01:
            sample = (sample + 1) / 2
        
        return sample
        
    
    def forward(self, x):
        
        out = self.sdedit(x, self.t)# [0, 1]
        out = self.classifier(out)
        return out
    


def generate_x_adv_denoised(x, y, diffusion, model, classifier, pgd_conf, device, t):
    
    
    net = Denoised_Classifier(diffusion, model, classifier, t)
    
    
    adversary = LinfPGDAttack(  net,
                                loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"), 
                                eps=pgd_conf['eps'],
                                nb_iter=pgd_conf['iter'], 
                                eps_iter=pgd_conf['alpha'], 
                                rand_init=True, 
                                targeted=False
                                )
    
    x_adv = adversary.perturb(x, y)
    
    return x_adv

@torch.no_grad()
def generate_x_adv_denoised_v2(x, y, diffusion, model, classifier, pgd_conf, device, t):
    
    
    net = Denoised_Classifier(diffusion, model, classifier, t)

    
    delta = torch.zeros(x.shape).to(x.device)
    # delta.requires_grad_()

    loss_fn=torch.nn.CrossEntropyLoss(reduction="sum")

    eps = pgd_conf['eps']
    alpha = pgd_conf['alpha']
    iter = pgd_conf['iter']

    

    for pgd_iter_id in range(iter):
        
        x_diff = net.sdedit(x+delta, t).detach()

        x_diff.requires_grad_()

        with torch.enable_grad():

            loss = loss_fn(classifier(x_diff), y)

            loss.backward()

            grad_sign = x_diff.grad.data.sign()

        delta += grad_sign * alpha

        delta = torch.clamp(delta, -eps, eps)
    print("Done")

    x_adv = torch.clamp(x+delta, 0, 1)    
    return x_adv.detach()



def Attack_Global(classifier, device, respace, t, eps=16, iter=10, name='attack_global', alpha=2, version='v1'):
    
    
    pgd_conf = gen_pgd_confs(eps=eps, alpha=alpha, iter=iter, input_range=(0, 1))

    save_path = f'vis/{name}_{version}/{classifier}_eps{eps}_iter{iter}_{respace}_t{t}/'

    mp(save_path)


    
    classifier = get_archs(classifier, 'imagenet')
    
    classifier = classifier.to(device)
    classifier.eval()
    
    dataset = get_dataset(
        'imagenet', split='test'
    )
    

    model, diffusion = get_imagenet_dm_conf(device=device, respace=respace)
    
    skip = 200
    c = 0

    for i in tqdm(range(dataset.__len__())):
        if i % skip != 0:
            continue
        time_st = time.time()
        print(f'{c}/{dataset.__len__()//skip}')


        x, y = dataset[i]
        x = x[None, ].to(device)
        y = torch.tensor(y)[None, ].to(device)
        
        y_pred = classifier(x).argmax(1) # original prediction

        if version == 'v1':
            x_adv = generate_x_adv_denoised(x, y_pred, diffusion, model, classifier, pgd_conf, device, t)
        elif version == 'v2':
            x_adv = generate_x_adv_denoised_v2(x, y_pred, diffusion, model, classifier, pgd_conf, device, t)

        cprint('time: {:.3}'.format(time.time() - time_st), 'g')

        with torch.no_grad():
        
            net = Denoised_Classifier(diffusion, model, classifier, t)
        
            pred_x0 = net.sdedit(x_adv, t)

        pkg = {
            'x': x,
            'y': y,
            'x_adv': x_adv,
            'x_adv_diff': pred_x0,
        }



        print(x_adv.min(), x_adv.max(), (x-x_adv).abs().max())

        
        torch.save(pkg, save_path+f'{i}.bin')
        si(torch.cat([x, x_adv, pred_x0], -1), save_path + f'{i}.png')
        print(y_pred, classifier(x_adv).argmax(1), classifier(pred_x0).argmax(1))


        

        c += 1

# Attack_Global('resnet101', 1, 'ddim100', t=2, eps=16, iter=1)

# Attack_Global('resnet50', 3, 'ddim100', t=2, eps=16, iter=1)
# Attack_Global('resnet50', 3, 'ddim100', t=2, eps=16, iter=2)
# Attack_Global('resnet50', 3, 'ddim100', t=2, eps=16, iter=5)
# Attack_Global('resnet50', 3, 'ddim100', t=2, eps=16, iter=10)
# Attack_Global('resnet50', 3, 'ddim100', t=3, eps=16, iter=10)


# eps = 8 #####################
# Attack_Global('resnet50', 1, 'ddim50', t=2, eps=8, iter=1, name='attack_global_new2')
# Attack_Global('resnet50', 1, 'ddim50', t=2, eps=8, iter=2, name='attack_global_new2')
# Attack_Global('resnet50', 1, 'ddim50', t=2, eps=8, iter=5, name='attack_global_new2')
# Attack_Global('resnet50', 1, 'ddim50', t=2, eps=8, iter=10, name='attack_global_new2')


# Attack_Global('resnet50', 1, 'ddim100', t=2, eps=8, iter=1, name='attack_global_new2')
# Attack_Global('resnet50', 1, 'ddim100', t=2, eps=8, iter=2, name='attack_global_new2')
# Attack_Global('resnet50', 1, 'ddim100', t=2, eps=8, iter=5, name='attack_global_new2')
# Attack_Global('resnet50', 1, 'ddim50', t=2, eps=8, iter=10, name='attack_global_new2')


# eps=32 ######################
Attack_Global('resnet50', 0, 'ddim50', t=3, eps=16, iter=10, name='attack_global_gradpass', alpha=2, version='v2')
# Attack_Global('resnet50', 0, 'ddim50', t=3, eps=16, iter=10, name='attack_global_gradpass', alpha=2)

# Attack_Global('resnet50', 0, 'ddim40', t=3, eps=16, iter=10, name='attack_global_gradpass', alpha=4)



# Attack_Global('resnet50', 5, 'ddim10', t=2, eps=32, iter=10, name='attack_global_new2')
# Attack_Global('resnet50', 1, 'ddim40', t=2, eps=16, iter=10, name='attack_global_new2')
# Attack_Global('resnet50', 1, 'ddim30', t=2, eps=16, iter=10, name='attack_global_new2')

# Attack_Global('resnet50', 1, 'ddim30', t=2, eps=16, iter=10, name='attack_global_new')
# Attack_Global('resnet50', 1, 'ddim20', t=2, eps=16, iter=10, name='attack_global_new')

