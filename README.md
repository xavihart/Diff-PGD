<div align="center">

<h2>Diffusion-Based Adversarial Sample Generation for Improved Stealthiness and Controllability [NeurIPS'2023]</h2>

**[Haotian Xue <sup>1](https://xavihart.github.io/), [Alexandre Araujo <sup>2](https://www.alexandrearaujo.com/), [Bin Hu <sup>3](https://binhu7.github.io/), and [Yongxin Chen <sup>1](https://yongxin.ae.gatech.edu/)**


<sup>1</sup> GaTech, <sup>2</sup> NYU, <sup>3</sup> UIUC

</div>


![](figures/pull_figures.png)

## Introduction

Diff-PGD utilizes strong prior knowledge of Diffusion Model to generate adversarial samples with higher steathiness and controllability. Diff-PGD has the following edges:
- Generate adversarial samples with higher **steathiness** and **controllability** 
- Can be easily applied to different scenarios (e.g. global, region, style-based, physical world)
- Higher **transferability** and **anti-purification** power



## Content
- [Introduction](#introduction)
- [News](#news-)
- [Envs](#envs)
- [Run Global Attack](#run-global-attack)
- [Run Regional Attack](#run-regional-attack)
- [Run Style-based Attack](#run-style-based-attack)
- [Run Physical World Attack](#run-physical-world-attack)
- [Some other useful code descriptions](#some-other-useful-code-descriptions)


## News

:star: [2023-09-21] Diff-PGD is accepted by NeurIPS 2023!

:star: [2023-09-21] Accelerated version (v2) can be now used

:star: [2023-06-03] We release the code!

:star: [2023-05-25] Our paper is released:  https://arxiv.org/abs/2305.16494




## Envs

1. Creat conda env using `env.yml`:
```
conda env create -f env.yml
```
2. Activate conda env:
```
source activate diff-pgd
```
3. Download [[DM checkpoint]](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt) and put them into `ckpt/`


4. Prepare imagenet dataset, and in `code/dataset.py:17`, replace it with:
```
os.environ['IMAGENET_LOC_ENV'] = YOUR_IMAGENET_PATH
```

## Run Global Attack

Here we provide an example to run gobal attack for imagenet val set:

```
python code/attack_global.py
```

or you can import `Attack_Global` from `code/attack_global.py` to build your customized settings:

```
Attack_Global(classifier, device, respace, t, eps, iter, name, alpha, version, skip)

[classifier]:                                       target classifier to attack
[device]:                                                        device, gpu id
[respace]                              diffusion model accelerator, e.g. ddim10
[t]                                                      reverse step in SDEdit
[eps]                                                       l_inf budget of PGD
[iter]                                                 iterations number of PGD
[name]                                                          name of the exp
[alpha]                                                        step size of PGD
[version]                base version v1, v2 is a faster version to be released 
[skip]                                                 skip interval of dataset
``` 



## Run Regional Attack


Here we provide an example to run gobal attack for imagenet val set:

```
python code/attack_region.py
```

similarly, we have the settings as:

```
Attack_Region(classifier, device, respace, t, eps, iter, name, skip, ratio):

[classifier]:                                       target classifier to attack
[device]:                                                        device, gpu id
[respace]                              diffusion model accelerator, e.g. ddim10
[t]                                                      reverse step in SDEdit
[eps]                                                       l_inf budget of PGD
[iter]                                                 iterations number of PGD
[name]                                                          name of the exp
[alpha]                                                        step size of PGD
[skip]                                                 skip interval of dataset
[ratio]                   ratio for masked square region, size ratio*H, ratio*W
```




## Run Style-based Attack


We provide some examples of style-based adversarial attack (mostly from AdvCAM official repo) in `data/advcam`, include target image, style reference image and segmentation masks.

In the code example `code/attack_style.py`, we define some scenes like `['tra4', 'tra1', 'tra5', 'tra6', 'leaf', 'car', 'lamp', 'car']`, we also encourage users to define their own stlye-based attack settings.

```
python code/attack_style.py
```

the core function has settings like:

```
Attack_Region_Style(exp_name, classifier, device, respace, t, eps, iter, name):


[exp_name]:                       defined name of example, e.g. tra5, tra4, car
[classifier]:                                       target classifier to attack
[device]:                                                        device, gpu id
[respace]                              diffusion model accelerator, e.g. ddim10
[t]                                                      reverse step in SDEdit
[eps]                                                       l_inf budget of PGD
[iter]                                                 iterations number of PGD
[name]                                                          name of the exp
```

## Run Physical World Attack

We provide some physical world attack data in `data/our_dataset`, including background and pathes in the paper:

- backbag attack background: `data/our_dataset/bag`
- computer mouse attack background: `dataset/our_dataset/computer_mouse`
- patches and masks: `data/our_dataset/patches`

to run example code:

```
python code/attack_physics.py
```

core function and settings for computer mouse as an example:
```
Attack_Physics(mode='diff-pgd', bkg_name='computer_mouse', patch_name='apple', classifier='resnet50', device=0, respace='ddim10', t=2, target=None, c_w=1, s_m=0.01, iter=4000, name='attack_physics')
```


## Some other useful code descriptions

- `code/guided_diffusion/` : official implementation of guided_diffusion
-  `code/dataset.py`: load imagenet dataset
-  `code/load_dm.py`: load diffusion model used in Diff-PGD
-  `code/attack_tools.py`: some adversarial attack tools
-  `code/stat.py`: run statistics on success rate, transferability and anti-purification power [to be released]




## Cited as:

```
@article{xue2023diffusion,
  title={Diffusion-Based Adversarial Sample Generation for Improved Stealthiness and Controllability},
  author={Xue, Haotian and Araujo, Alexandre and Hu, Bin and Chen, Yongxin},
  journal={arXiv preprint arXiv:2305.16494},
  year={2023}
}
```
