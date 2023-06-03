<div align="center">

<h2>Diffusion-Based Adversarial Sample Generation for Improved Stealthiness and Controllability</h2>

**[Haotian Xue <sup>1](https://xavihart.github.io/), [Alexandre Araujo <sup>2](https://www.alexandrearaujo.com/), [Bin Hu <sup>3](https://binhu7.github.io/), and [Yongxin Chen <sup>1](https://yongxin.ae.gatech.edu/)**


<sup>1</sup> GaTech, <sup>2</sup> NYU, <sup>3</sup> UIUC

</div>


![](figures/pull_figures.png)

TL;DR: PGD + Diffusion Model = stronger adversarial attack

## Content
- [News](#news)
- [Introduction](#introduction)
- [Envs](#envs)
- [Run Global Attack](#run-global-attack)
- [Run Regional Attack](#run-regional-attack)
- [Run Style-based Attack](#run-style-based-attack)
- [Run Physical World Attack](#run-physical-world-attack)
- [Updates](#todo)

## News:

:star: [2023-06-03] Faster V2 will be released soon

:star: [2023-06-03] We release the code!

:star: [2023-05-25] Our paper is released:  https://arxiv.org/abs/2305.16494


## Introduction

Diff-PGD utilizes strong prior knowledge of Diffusion Model to generate adversarial samples with higher steathiness and controllability. Diff-PGD has the following edges:
- Generate adversarial samples with higher **steathiness** and **controllability** 
- Can be easily applied to different scenarios (e.g. global, region, style-based, physical world)
- Higher **transferability** and **anti-purification** power




## Envs

1. Creat conda env using `env.yml`:
```
conda env create -f environment.yml
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

Here we provide a code snipplet to run gobal attack:

```
python code/attack_global.py
```

or you can import `Attack_Global` from `code/attack_global.py` to build your customized settings:

```
Attack_Global(classifier, device, respace, t, eps, iter, name, alpha version)

[classifier]:                                       target classifier to attack
[device]:                                                        device, gpu id
[respace]                              diffusion model accelerator, e.g. ddim10
[t]                                                      reverse step in SDEdit
[eps]                                                       l_inf budget of PGD
[iter]                                                 iterations number of PGD
[name]                                                          name of the exp
[alpha]                                                        step size of PGD
[version]                base version v1, v2 is a faster version to be released 

``` 

## Run Regional Attack



## Run Style-based Attack


## Run Physical World Attack


## TODO
- [x] Paper is out

- [ ] Code for global attack

- [ ] Code for style-based attack

- [ ] Code for physical-world attack

- [ ] Publish pip version




