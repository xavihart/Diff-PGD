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

def get_model_list(model_name_list, device):
    model_list = []
    for name in model_name_list:
        net = get_archs(name)
        net = net.to(device)
        net.eval()
        model_list.append(net)
    return model_list


def transfer_bench(adv_sample, clean_sample, device, model_list):
    r = []
    for net in model_list:
        
        pred = net(adv_sample).argmax(1)
        pred_clean = net(clean_sample).argmax(1)
        if pred_clean == pred:
            r.append(0) # failed to change the decision of network
        else:
            r.append(1)
    return torch.tensor(r)



class Denoised_Classifier(torch.nn.Module):
    def __init__(self, diffusion, model, classifier, t):
        super().__init__()
        self.diffusion = diffusion
        self.model = model
        self.classifier = classifier
        self.t = t
    
    def sdedit(self, x, t, to_01=True):

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
        
        out = self.sdedit(x, self.t) # [0, 1]
        out = self.classifier(out)
        return out
    

def purify_bench(adv_sample, model, d_classifier, classifier, n, y_pred):
    a = 0
    for i in n:
        y_pred_0 = d_classifier(adv_sample).argmax(1)
        if y_pred == y_pred_0:
            a += 0
    return a
        
    
