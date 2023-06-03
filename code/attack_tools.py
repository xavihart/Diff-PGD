


import random
import torch
import os
from advertorch.attacks import LinfPGDAttack
import torchvision.models as models
# from torchvision.models import VGG19_Weights
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from scipy.ndimage import rotate
import math

def gen_pgd_confs(eps, alpha, iter, input_range=(-1, 1)):
    conf = {}
    conf['eps'] = eps / 255.0 * (input_range[1] - input_range[0])
    conf['alpha'] = alpha / 255.0 * (input_range[1] - input_range[0])
    conf['iter'] = iter
    return conf

def gen_mask(x, type, ratio):
    b, c, h, w = x.shape
    if type == 'square':
        mask = torch.ones_like(x)
        m_h, m_w =  int(h * ratio), int(w * ratio)
        for b in range(b):
            # mask mask[b]
            x_s = random.randint(0, h - m_h)
            y_s = random.randint(0, w - m_w)
            mask[b][:, x_s:x_s+m_h, y_s:y_s+m_w] = 0
    return mask

def gen_pgd_sample(net, x, iter=10, eps=32, alpha=2):
    adversary = LinfPGDAttack(net, loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"), 
                                  eps=eps/255,
                                  nb_iter=iter, 
                                  eps_iter=alpha/255, 
                                  rand_init=False, 
                                  targeted=False
                                )
    y = net(x).argmax(1)
    return adversary.perturb(x, y)


class wrapper(torch.nn.Module):
        def __init__(self, net, x_masked, mask):
            super().__init__()
            self.net = net
            self.x_masked = x_masked
            self.mask = mask
        def forward(self, x_unmasked):
            return self.net(x_unmasked*self.mask + self.x_masked)
        

def gen_region_pgd_sample(net, x, region_mask, iter=10, eps=16, alpha=2):
    
    
    x_unmasked = x * region_mask
    x_masked   = x.detach() * (1 - region_mask)

        
    net = wrapper(net, x_masked, region_mask)

  
    adversary = LinfPGDAttack(net, loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"), 
                                  eps=eps/255,
                                  nb_iter=iter, 
                                  eps_iter=alpha/255, 
                                  rand_init=False, 
                                  targeted=False
                                )
    y = net(x_masked + x_unmasked).argmax(1)



    result = region_mask * adversary.perturb(x_unmasked, y) + x_masked

    return result












# style loss

class ContentLoss(nn.Module):
    
    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        self.target = target.detach() * weight
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion(input * self.weight, self.target)
        self.output = input
        return self.output

    def backward(self, retain_variables=True):
        self.loss.backward(retain_graph=retain_variables)
        return self.loss

class GramMatrix(nn.Module):
    
    def forward(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    
    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.output = input.clone()
        self.G = self.gram(input)
        self.G.mul_(self.weight)
        self.loss = self.criterion(self.G, self.target)
        return self.output

    def backward(self, retain_variables=True):
        self.loss.backward(retain_graph=retain_variables)
        return self.loss


class Normalization(torch.nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize ``img``
        return (img - self.mean) / self.std
    
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def get_style_model_and_losses(style_img, content_img,
                               style_weight=1000, content_weight=1,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    
    cnn = models.vgg19(pretrained=True).features
    cnn = cnn.to(content_img.device)
    cnn.eval()

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    model = nn.Sequential()  # the new Sequential module network
    gram = GramMatrix()  # we need a gram module in order to compute style targets

    # move these modules to the GPU if possible:
    # if use_cuda:
    #     model = model.cuda()
    #     gram = gram.cuda()

    i = 1
    for layer in list(cnn):
        if isinstance(layer, nn.Conv2d):
            name = "conv_" + str(i)
            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)

        if isinstance(layer, nn.ReLU):
            name = "relu_" + str(i)
            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)

            i += 1

        if isinstance(layer, nn.MaxPool2d):
            name = "pool_" + str(i)
            model.add_module(name, layer)  # ***

    return model, style_losses, content_losses




def get_style_loss_grad(x, model, style_losses, content_losses, style_weight=100, content_weight=1):
    
    assert x.requires_grad==True
    x.retain_grad()
    
    
    model(x)

    style_score = 0
    content_score = 0

    for sl in style_losses:
        style_score += sl.backward()
    for cl in content_losses:
        content_score += cl.backward()

    score = style_score + content_score
    # score.backward()
    

    return x.grad





def init_patch_circle(image_size, patch_size):
    image_size = image_size**2
    noise_size = int(image_size*patch_size)
    radius = int(math.sqrt(noise_size/math.pi))
    patch = np.zeros((1, 3, radius*2, radius*2))    
    for i in range(3):
        a = np.zeros((radius*2, radius*2))    
        cx, cy = radius, radius # The center of circle 
        y, x = np.ogrid[-radius: radius, -radius: radius]
        index = x**2 + y**2 <= radius**2
        a[cy-radius:cy+radius, cx-radius:cx+radius][index] = np.random.rand()
        idx = np.flatnonzero((a == 0).all((1)))
        a = np.delete(a, idx, axis=0)
        patch[0][i] = np.delete(a, idx, axis=1)
    return patch, patch.shape


# physical attacks
def circle_transform(patch, data_shape, patch_shape, image_size):
    # get dummy image 
    x = np.zeros(data_shape)
   
    # get shape
    m_size = patch_shape[-1]
    
    for i in range(x.shape[0]):

        # random rotation
        rot = np.random.choice(360)
        for j in range(patch[i].shape[0]):
            patch[i][j] = rotate(patch[i][j], angle=rot, reshape=False)
        for j in range(patch[i].shape[0]):
            patch[i][j] = np.rot90(patch[i][j], rot)
        
        # random location
        random_x = np.random.choice(image_size)
        if random_x + m_size > x.shape[-1]:
            while random_x + m_size > x.shape[-1]:
                random_x = np.random.choice(image_size)
        random_y = np.random.choice(image_size)
        if random_y + m_size > x.shape[-1]:
            while random_y + m_size > x.shape[-1]:
                random_y = np.random.choice(image_size)
       
        # apply patch to dummy image  
        x[i][0][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[i][0]
        x[i][1][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[i][1]
        x[i][2][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[i][2]
    
    mask = np.copy(x)
    mask[mask != 0] = 1.0
    
    return x, mask, patch.shape

