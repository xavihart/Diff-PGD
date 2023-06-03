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





    
def ddim_sample_self_made(model, diffusion, shape, device):

    # shape: b * c * h * w
    B, C, H, W = shape



    indices = list(range(diffusion.num_timesteps))[::-1]

    # last step sample

    sample = torch.randn(*shape, device=device)

    from tqdm.auto import tqdm

    indices = tqdm(indices)

    for id in indices:
        
        t = torch.full((B, ), id).long().to(device)

        sample = diffusion.ddim_sample(model, sample, t)["sample"]
    

    return sample



    

def recons_self_made(diffusion, model, image, mask, device, ddim, scale=2):
    # image : bchw, mask : bchw, 0, 1
    B, C, H, W = image.shape


    indices = list(range(diffusion.num_timesteps))[::-1]

    # last step sample

    sample = torch.randn(image.shape, device=device)


    indices = tqdm(indices)

    

    for id in indices:
        
        t = torch.full((B, ), id).long().to(device)

        with torch.no_grad():

            if ddim:
                sample = diffusion.ddim_sample_reconstruct(model, sample, t, image, mask, scale)["sample"]
            else:
                pass
                # sample = diffusion.p_sample(model, sample, t)["sample"]

        sample = sample * (1. - mask) + diffusion.q_sample(image, t) * mask
    

    return sample
    
    





def repaint_self_made(diffusion, model, image, mask, device, ddim, resample_num=10, jump=10):
    # image : bchw, mask : bchw, 0, 1
    B, C, H, W = image.shape


    indices = list(range(diffusion.num_timesteps))[::-1]

    # last step sample

    sample = torch.randn(image.shape, device=device)


    indices = tqdm(indices)

    

    for id in indices:
        
        if id % jump:
            resample_num_ = resample_num
        else:
            resample_num_ = 1
        
        t = torch.full((B, ), id).long().to(device)

        with torch.no_grad():

            if resample_num_ == 1:

                if ddim:
                    sample = diffusion.ddim_sample(model, sample, t)["sample"]
                else:
                    sample = diffusion.p_sample(model, sample, t)["sample"]
            
            else:
                if ddim:
                    sample = diffusion.ddim_sample(model, sample, t)["pred_xstart"]
                else:
                    sample = diffusion.p_sample(model, sample, t)["pred_xstart"]
                
                for _ in range(resample_num_-2):
                    sample = diffusion.q_sample(sample, t)
                    if ddim:
                        sample = diffusion.ddim_sample(model, sample, t)["pred_xstart"]
                    else:
                        sample = diffusion.p_sample(model, sample, t)["pred_xstart"]
                
                sample = diffusion.q_sample(sample, t)
                
                if ddim:
                    sample = diffusion.ddim_sample(model, sample, t)["sample"]
                else:
                    sample = diffusion.p_sample(model, sample, t)["sample"]
                
                    



        sample = sample * (1. - mask) + diffusion.q_sample(image, t) * mask
    
    return sample


def test_dm():
    dataset = get_dataset(
        'imagenet', split='test'
    )
    x, _ = dataset[0]
    x = x.cuda()[None,...]
    
    x = x * 2 - 1
    
    # print(x.min(), x.max())
    
    model, diffusion = get_imagenet_dm_conf(device=x.device, respace='ddim100')
    
    
    t = torch.full((1, ), 10).long().cuda()
    
    x_t = diffusion.q_sample(x, t) 
    
    # print(x_t.min(), x_t.max())
    
    si(x_t, 'vis/noised_x.png', to_01=True)
    
    out = diffusion.ddim_sample(model, x_t, t)['pred_xstart']
    print(out.min(), out.max())
    
    si(out, 'vis/ddim_pred_x0.png',  to_01=True)
    
    # out = diffusion.ddim_sample_loop(model, shape=(10, 3, 256, 256), progress=True)
    # torch.save(out, 'ddim_sample.bin')
    with torch.no_grad():
        out = ddim_sample_self_made(model, diffusion, (10, 3, 256, 256), 0)

    si(out, 'vis/ddim_sample_x0.png', to_01=True)




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

def test_repaint():
    dataset = get_dataset(
        'imagenet', split='test'
    )
    
    x = torch.stack([dataset[i][0] for i in range(0, 10000, 1000)])

    print(x.shape)
    
    x = x * 2 - 1
    x = x.to(0)
    
    mask = gen_mask(x, type='square', ratio=0.4)

    si(mask, 'vis/mask.png')
    respace='ddim100'
    model, diffusion = get_imagenet_dm_conf(device=x.device, respace=respace)
    out = repaint_self_made(diffusion, model, x, mask, device=x.device, ddim=True if 'ddim' in respace else False)
    si(out, f'vis/repaint_{respace}.png', to_01=True)
    si(x*mask, f'vis/before_repaint_{respace}.png', to_01=True)

    
    return

def test_reconstruct(device, ratio):
    dataset = get_dataset(
        'imagenet', split='test'
    )
    
    x = torch.stack([dataset[i][0] for i in range(0, 10000, 1000)])

    print(x.shape)
    
    x = x * 2 - 1
    x = x.to(device)
    
    mask = gen_mask(x, type='square', ratio=ratio)

    respace='ddim50'
    model, diffusion = get_imagenet_dm_conf(device=x.device, respace=respace)
    out = recons_self_made(diffusion, model, x, mask, device=x.device, ddim=True if 'ddim' in respace else False)
    si(out, f'vis/recons_{respace}.png', to_01=True)
    si(x*mask, f'vis/before_recons_{respace}.png', to_01=True)

    
    return
    
    





def play_with_pgd(classifier, device):
    
    pgd_conf = gen_pgd_confs(eps=4, alpha=1, iter=100, input_range=(0, 1))
    
    classifier = get_archs(classifier, 'imagenet')
    
    classifier = classifier.to(device)
    classifier.eval()
    
    dataset = get_dataset(
        'imagenet', split='test'
    )
    
    skip = 200
    
    a = 0
    c = 0
    
    
    transfer_list = torch.zeros(len(IMAGENET_MODEL))
    
    c = 0
    
    for i in range(dataset.__len__()):
        if i % skip != 0:
            continue
        print(f'{c}/{dataset.__len__()//skip}')
        x, y = dataset[i]
        x = x[None, ].to(device)
        y = torch.tensor(y)[None, ].to(device)
            

        pred_raw = classifier(x).argmax(1)
        
        adversary = LinfPGDAttack(classifier, loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"), 
                                  eps=pgd_conf['eps'],
                                  nb_iter=pgd_conf['iter'], 
                                  eps_iter=pgd_conf['alpha'], 
                                  rand_init=True, 
                                  clip_min=0.0, 
                                  clip_max=1.0,
                                  targeted=False
                                )
        
        x_adv = adversary.perturb(x, y)
        # print(time.time() - t)
        
        si(x_adv, 'vis/x_adv.png')
        
        pred = classifier(x_adv).argmax(1)
        print('gt:[{}], raw:[{}], adv:[{}]'.format(y[0].item(), pred_raw[0].item(), pred[0].item()))
        
        transfer_r = transfer_bench(IMAGENET_MODEL, x_adv, x, y, device)
        
        transfer_list += torch.tensor(transfer_r)
        
        c += 1
    
    plt.bar(IMAGENET_MODEL, transfer_list/c)
    plt.savefig(f"vis/transfer_bench/transfer_bench_eps{pgd_conf['eps']}.png")
        
        
        


def loss_fn():
    pass




        

def generate_x_adv_denoised(x, y, diffusion, model, classifier, pgd_conf, device, t):
    
    
    net = Denoised_Classifier(diffusion, model, classifier, t)
    
    
    adversary = LinfPGDAttack(net, loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"), 
                                  eps=pgd_conf['eps'],
                                  nb_iter=pgd_conf['iter'], 
                                  eps_iter=pgd_conf['alpha'], 
                                  rand_init=True, 
                                  targeted=False
                                )
    
    x_adv = adversary.perturb(x, y)
    
    return x_adv
    

    
def exp_1(classifier, device, respace, t, eps=8, iter=10, ):
    
    
    pgd_conf = gen_pgd_confs(eps=eps, alpha=1, iter=iter, input_range=(0, 1))

    save_path = f'vis/exp1/{classifier}_eps{eps}_iter{iter}_{respace}_t{t}/'

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

    for i in range(dataset.__len__()):
        if i % skip != 0:
            continue
        time_st = time.time()
        print(f'{c}/{dataset.__len__()//skip}')


        x, y = dataset[i]
        x = x[None, ].to(device)
        y = torch.tensor(y)[None, ].to(device)
        
        y_pred = classifier(x).argmax(1)
            
        x_adv = generate_x_adv_denoised(x, y_pred, diffusion, model, classifier, pgd_conf, device, t)
        
        net = Denoised_Classifier(diffusion, model, classifier, t)
        
        pred_x0 = net.sdedit(x_adv, t)

        pkg = {
            'x': x,
            'y': y,
            'x_adv': x_adv,
            'x_adv_diff': pred_x0,
        }

        
        torch.save(pkg, save_path+f'{i}.bin')
        si(torch.cat([x, x_adv, pred_x0], -1), save_path + f'{i}.png')


        cprint('time: {:.3}'.format(time.time() - time_st), 'g')

        c += 1

        
@torch.enable_grad()       
def pgd_baseline_sample(net, x, iter=10, eps=8, alpha=2):
    adversary = LinfPGDAttack(net, loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"), 
                                  eps=eps/255,
                                  nb_iter=iter, 
                                  eps_iter=alpha/255, 
                                  rand_init=True, 
                                  targeted=False
                                )
    y = net(x).argmax(1)
    return adversary.perturb(x, y)
    
        

def diff_camouflage_v1(x, y, diffusion, model, classifier, pgd_conf, device, t):
    
    pass      



def plot_multi_bars(label_list, value_list, p, sample_labels):
    label_num = len(label_list)
    sample_num = len(value_list)
    inds = np.arange(label_num)
    w = 0.1
    colors = get_plt_color_list()
    bars = []
    for i, v in enumerate(value_list):
        bar = plt.bar(inds+w*i, v, w, color=colors[i])
        bars.append(bar)
    plt.xticks(inds+w*int(sample_num/2), label_list)
    plt.legend(bars, sample_labels)
    plt.savefig(p)


# plot_multi_bars(['a', 'b', 'c'],  [torch.tensor([1,2,3]), [3,4,5], [4,5,6], [4,5,6], [5,6,7]], 'bar.png', ['x', 'x_adv', 'x_daa', 'x2', 'x3'])

@torch.no_grad()
def cal_transfer_bench(p, classifier, device, out_p):
    net = get_archs(classifier)
    net = net.to(device)
    net.eval()
    bins = glob.glob(p+'/*.bin')
    
    def get_pred(x, net):
        return net(x).argmax(1)[0].item()    
    

    x_adv_list = torch.zeros(len(IMAGENET_MODEL))
    x_adv_diff_list = torch.zeros(len(IMAGENET_MODEL))
    x_pgd_list = torch.zeros(len(IMAGENET_MODEL))

    i = 0
    for bin in tqdm(bins):
        i+=1
        # if i > 5:
        #     break
        # print(bin)
        pkg = torch.load(bin)
        x = pkg['x'].to(device)
        x_adv = pkg['x_adv'].to(device)
        x_adv_diff = pkg['x_adv_diff'].to(device)
        x_pgd = pgd_baseline_sample(net, x, eps=16, iter=10, alpha=2)

        x_adv_r = transfer_bench(IMAGENET_MODEL, x_adv, x, device)
        x_adv_diff_r = transfer_bench(IMAGENET_MODEL, x_adv_diff, x, device)
        x_pgd_r = transfer_bench(IMAGENET_MODEL, x_pgd, x, device)

        x_adv_list += torch.tensor(x_adv_r)
        x_adv_diff_list += torch.tensor(x_adv_diff_r)
        x_pgd_list += torch.tensor(x_pgd_r)

        print(x_adv_r)
        print(x_adv_diff_r)
        print(x_pgd_r)
    
    print(x_adv_list)
    print(x_adv_diff_list)
    print(x_pgd_list)

    plot_multi_bars(IMAGENET_MODEL, [x_adv_list, x_adv_diff_list, x_pgd_list], f'vis/{out_p}.png', ['x_adv', 'x_diff', 'x_pgd'])



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


def purify(x, t, model, diffusion):
    p_net = Denoised_Classifier(diffusion, model, classifier=None, t=t)
    return p_net.sdedit(x, t)



from bench import *

@torch.no_grad()
def cal_anti_purify_bench(p, classifier, device, out_p, respace='ddim50', t=5, save_p='draw/antip/'):
    
    mp(save_p)
    
    net = get_archs(classifier)
    net = net.to(device)
    net.eval()
    model, diffusion = get_imagenet_dm_conf(respace=respace, device=device)

    bins = glob.glob(p+'/*.bin')
    
    def get_pred(x, net):
        return net(x).argmax(1)[0].item()    
    

    x_adv_list = torch.zeros(len(IMAGENET_MODEL))
    x_adv_diff_list = torch.zeros(len(IMAGENET_MODEL))
    x_pgd_list = torch.zeros(len(IMAGENET_MODEL))

    i = 0
    for bin in tqdm(bins):
        i+=1
        # if i > 2:
        #     break
        print(bin)
        pkg = torch.load(bin)
        x = pkg['x'].to(device)

        x_adv = pkg['x_adv'].to(device)
        x_adv_diff = pkg['x_adv_diff'].to(device)
        x_pgd = pgd_baseline_sample(net, x, eps=16, iter=10, alpha=2)

        x_adv_p = purify(x_adv, t, model, diffusion)
        x_adv_diff_p = purify(x_adv_diff, t, model, diffusion)
        x_pgd_p = purify(x_pgd, t, model, diffusion)


        # x_adv_r = transfer_bench(IMAGENET_MODEL, x_adv_p, x, device)
        # x_adv_diff_r = transfer_bench(IMAGENET_MODEL, x_adv_diff_p, x, device)
        # x_pgd_r = transfer_bench(IMAGENET_MODEL, x_pgd_p, x, device)

        scale=5
        si(torch.cat(
            [
             torch.cat([x_pgd, x_adv_diff], -1),
             torch.cat([x_pgd_p, x_adv_diff_p], -1),
             torch.cat([(x_pgd_p-x_pgd)*scale, (x_adv_diff_p-x_adv_diff)*scale], -1)
            # torch.cat([x_pgd, x_adv, x_adv_diff], -1),
            # torch.cat([x_pgd_p, x_adv_p, x_adv_diff_p], -1),
            # torch.cat([(x_pgd_p-x_pgd)*scale, (x_adv_p-x_adv)*scale, (x_adv_diff_p-x_adv_diff)*scale], -1)
            ],
            -2
        ), save_p+f'{i}.png')
        
        print(save_p+f'{i}.png')
        print(torch.abs((x_pgd_p-x_pgd)).norm(p=2),
              torch.abs((x_adv_p--x_adv)).norm(p=2),
              torch.abs((x_adv_diff_p-x_adv_diff)).norm(p=2))

        # x_adv_list += torch.tensor(x_adv_r)
        # x_adv_diff_list += torch.tensor(x_adv_diff_r)
        # x_pgd_list += torch.tensor(x_pgd_r)

        # print(x_adv_r)
        # print(x_adv_diff_r)
        # print(x_pgd_r)
    
    print(x_adv_list)
    print(x_adv_diff_list)
    print(x_pgd_list)

    plot_multi_bars(IMAGENET_MODEL, [x_adv_list, x_adv_diff_list, x_pgd_list], f'{out_p}.png', ['x_adv', 'x_diff', 'x_pgd'])



def cal_stealthiness(image_list):
    pass


def diff_adv_patch(image, diffusion, model, classifier):
    pass
    

# test_dm()

# test_repaint()

# test_reconstruct(1, 0.2)

# play_with_pgd('resnet50', device=1)

# exp_1('resnet50', 2, 'ddim100', t=2)

# cal_transfer_bench('vis/exp1/resnet50_eps16_iter10_ddim100_t3', 'resnet50', 0, 'vis/transfer_bench.png')


# cal_anti_purify_bench('vis/attack_global_new2/resnet50_eps32_iter10_ddim10_t2', 'resnet50', 2, 'vis/anti_purify_bench', respace="ddim20", t=2)

cal_anti_purify_bench('vis/exp1/resnet50_eps16_iter10_ddim100_t2', 'resnet50', 2, 'vis/anti_purify_bench', respace="ddim20", t=2)
