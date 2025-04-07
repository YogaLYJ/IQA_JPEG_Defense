import torch
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
import copy
import os
import numpy as np
from hyperIQAclass import HyperIQA
import argparse


def r_robustness(preds_after,preds_befo,maxs=100,mins=0):
    change1 = preds_befo - mins
    change2 = maxs - preds_befo
    change_all = np.where(change1>change2,change1,change2)
    change_att = np.abs(preds_after-preds_befo)
    change_log = np.log10(change_all/change_att)
    change_log = np.mean(change_log)
    return change_log

def mse_loss(score_adv, score_tar):
    loss = torch.pow(score_adv - score_tar, 2)
    return loss

'''
FGSM--untargeted attack
x: input image
y: target score
'''
def IFGSM_IQA_untarget(model, x, org_score, eps=0.05, alpha=0.01, iteration=10, x_val_min=0, x_val_max=1):
        x_adv = Variable(x.data, requires_grad=True)
        for i in range(iteration):
            tmp_adv = norm(x_adv)
            score_adv = model(tmp_adv)
            if org_score > 50:
                loss = score_adv
            else:
                loss = - score_adv
            model.zero_grad()
            if x_adv.grad is not None:
                x_adv.grad.data.fill_(0)
            loss.backward(retain_graph=True)

            x_adv.grad.sign_()
            x_adv = x_adv - alpha*x_adv.grad
            x_adv = torch.where(x_adv > x+eps, x+eps, x_adv)
            x_adv = torch.where(x_adv < x-eps, x-eps, x_adv)
            x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
            x_adv = Variable(x_adv.data, requires_grad=True)

        score_org = model(norm(x))
        score_adv = model(norm(x_adv))

        return x_adv, score_adv, score_org

def fix_seed(seed):
    torch.manual_seed(seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    np.random.seed(seed)

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
transform_w_norm = transforms.Compose([
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))
])

transform_wo_norm = transforms.Compose([
    transforms.RandomCrop(size=224),
    transforms.ToTensor()
])

def norm(x):
    mean = torch.ones((3,224,224)).cuda()
    std = torch.ones((3,224,224)).cuda()
    mean[0,:,:]=0.485
    mean[1,:,:]=0.456
    mean[2,:,:]=0.406
    std[0,:,:]=0.229
    std[1,:,:]=0.224
    std[2,:,:]=0.225 
    
    x = (x - mean) / std
    
    return x

def de_norm(x):
    mean = torch.ones((3,224,224)).cuda()
    std = torch.ones((3,224,224)).cuda()
    mean[0,:,:]=0.485
    mean[1,:,:]=0.456
    mean[2,:,:]=0.406
    std[0,:,:]=0.229
    std[1,:,:]=0.224
    std[2,:,:]=0.225 
    
    x = x * std + mean
    
    return x

# ycx
df = pd.read_csv('./livec-test.csv')
images_all = df['filename']
mos_all = df['mos']
mini_list = [i for i in range(len(mos_all))]
images_mini = images_all
mos_mini = mos_all

img_folder = './test_data'

# save images without compression
def save(pert_image, path):
    pert_image = torch.round(pert_image * 255) / 255
    quantizer = transforms.ToPILImage()
    pert_image = quantizer(pert_image.squeeze())
    pert_image.save(path)
    
    return pert_image

# save images with compression
def save_jpeg(pert_image, path, jpeg_para):
    pert_image = torch.round(pert_image * 255) / 255
    quantizer = transforms.ToPILImage()
    pert_image = quantizer(pert_image.squeeze())
    pert_image.save(path, 'JPEG', quality=jpeg_para)
    
    return pert_image


def main(config):
    fix_seed(919)
    # load models
    use_cuda = True
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    if config.nt == 0:
        load_path = './checkpoints/livec_bs16_wo_nt.pth'
    elif config.nt == 0.001:
        load_path = './checkpoints/livec_bs16_nt_0.001.pth'
    else:
        raise ValueError("No such models with the NT weight being {}!".format(config.nt))
    print("Loaded model from {}.".format(load_path))
    model = HyperIQA(load_path).to(device)

    moses = []
    pred_scores = []
    pred_scores_ori = []
    eps = config.epsilon
    iter = config.step
    alpha = config.alpha
    patch = 25
    save_dir = './compressed_adv/IFGSM_eps{}_iter{}_nt{}'.format(eps, iter, config.nt)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    for i in range(len(mini_list)):
        score_i = []
        score_ori_i = []
        moses.append(mos_mini[i])
        # each image in the livec dataset is cropped into 25 patches
        for j in range(patch): 
            img_path = os.path.join(img_folder, images_mini[i][:-4]+'_'+str(j)+images_mini[i][-4:])
            image = pil_loader(img_path)
            image = transform_wo_norm(image)
            image = image.unsqueeze(0)
            image = image.cuda()
        
            original_score = model(norm(image)).detach().cpu().numpy()
            pert_image, pert_score, org_score = IFGSM_IQA_untarget(model, image, original_score, eps=eps, alpha=alpha, iteration=iter)
             
            # save adversarial examples without compression
            save_name = images_mini[i][:-4]+'_'+str(j)+'.bmp'
            save_bmp_dir = save_dir + '/BMP'
            if not os.path.exists(save_bmp_dir):
                os.mkdir(save_bmp_dir)
            save_path = os.path.join(save_bmp_dir, save_name)
            save(pert_image, save_path)
            
            # save adversarial examples with JPEG compression
            save_name = images_mini[i][:-4]+'_'+str(j)+'.jpeg'
            jpeg_para = config.compression
            save_jepg_dir = save_dir + '/JPEG{}'.format(jpeg_para)
            if not os.path.exists(save_jepg_dir):
                os.mkdir(save_jepg_dir)
            save_path = os.path.join(save_jepg_dir, save_name)
            save_jpeg(pert_image, save_path, jpeg_para)
                
            score_i.append(pert_score.detach().cpu().numpy())
            score_ori_i.append(org_score.detach().cpu().numpy())
        score_i = np.array(score_i)
        score_i = np.mean(score_i)
        pred_scores.append(score_i)
        score_ori_i = np.array(score_ori_i)
        score_ori_i = np.mean(score_ori_i)
        pred_scores_ori.append(score_ori_i)
        # print(score_i, score_ori_i)
        
    pred_scores = np.array(pred_scores).squeeze()
    moses = np.array(moses).squeeze()
    pred_scores_ori = np.array(pred_scores_ori).squeeze()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eps', dest='epsilon', type=float, default=0.005, help='the scale of FGSM attacks')
    parser.add_argument('--alp', dest='alpha', type=float, default=0.01, help='the scale of FGSM attacks')
    parser.add_argument('--step', dest='step', type=int, default=1, help='the number of steps in PGD')
    parser.add_argument('--nt_weight', dest='nt', type=float, default=0, help='the weight of the NT strategy')
    parser.add_argument('--jpeg_com', dest='compression', type=int, default=70, help='the parameter for the JPEG compression')

    config = parser.parse_args()
    main(config)
