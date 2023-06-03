from dataset import get_normalize_layer, get_input_center_layer
from torch.nn.functional import interpolate
import torchvision
from transformers import ViTFeatureExtractor, ViTModel, ViTForImageClassification, BeitFeatureExtractor, BeitForImageClassification
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from denoise import Denoiser

from robustbench.utils import load_model

IMAGENET_MODEL = [
    'resnet50',
    'resnet101',
    'resnet18',
    'wrn50',
    'wrn101',
]

IMAGENET_MODEL_ROBUST = [
    'Engstrom2019Robustness',
    'Salman2020Do_R50',
    'Salman2020Do_R18',
]


def get_archs(arch, dataset='imagenet'):
    if dataset == 'imagenet':
        if   arch == 'resnet50':
            model = torchvision.models.resnet50(pretrained=True)
        
        elif arch == 'resnet101':
            model = torchvision.models.resnet101(pretrained=True)
         
        elif arch == 'resnet18':
            model = torchvision.models.resnet18(pretrained=True)
            
        elif arch == 'wrn50':
            model = torchvision.models.wide_resnet50_2(pretrained=True)
            
        elif arch == 'wrn101':
            model = torchvision.models.wide_resnet101_2(pretrained=True)
            
        elif arch == 'beit':
            model = BeitForImageClassification.from_pretrained('microsoft/beit-large-patch16-224')

        elif arch in IMAGENET_MODEL_ROBUST:
            model = load_model(model_name=arch, dataset='imagenet', threat_model='Linf')
            return model
    
    normalize_layer = get_normalize_layer(dataset, vit=True if ("vit" in arch or 'beit' in arch) else False)
    
    
    return torch.nn.Sequential(normalize_layer, model)

    