import torch
import pandas as pd
from hyperIQAclass import HyperIQA
from scipy.stats import spearmanr, pearsonr, kendalltau
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import argparse

def fix_seed(seed):
    torch.manual_seed(seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    np.random.seed(seed)

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
def r_robustness(preds_after,preds_befo,maxs=100,mins=0):
    change1 = preds_befo - mins
    change2 = maxs - preds_befo
    change_all = np.where(change1>change2,change1,change2)
    change_att = np.abs(preds_after-preds_befo)
    change_log = np.log10(change_all/(change_att + 1e-6))
    change_log = np.mean(change_log)
    return change_log

transform_w_norm = transforms.Compose([
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))
])


def main(config):
    # load models
    use_cuda = True
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    if config.nt == 0:
        load_path = './checkpoints/livec_bs16_wo_nt.pth'
    elif config.nt == 0.001: # JPEG+NT defense
        load_path = './checkpoints/livec_bs16_nt_0.001.pth'
    else:
        raise ValueError("No such models with the NT weight being {}!".format(config.nt))
    print("Loaded model from {}.".format(load_path))
    model = HyperIQA(load_path).to(device)

    # data information
    df = pd.read_csv('./livec-test.csv')
    images_all = df['filename']
    moses_all = df['mos']
    img_org_folder = './test_data'
    img_adv_folder = './compressed_adv/IFGSM_eps0.005_iter1_nt0' # compressed adv images

    org_pred = []
    adv_pred_wo = [] # score without compression
    adv_pred_w = [] # score with compression
    mos = []
    for i in range(len(images_all)):
        score_org_i = []
        score_adv_i_wo = [] # score without compression
        score_adv_i_w = [] # score with compression
        mos.append(float(moses_all[i]))
        for j in range(25):
            img_org_path = os.path.join(img_org_folder,images_all[i][:-4] + '_' + str(j) + images_all[i][-4:])
            image_org = pil_loader(img_org_path)
            image_org = transform_w_norm(image_org)
            image_org = image_org.unsqueeze(0)
            image_org = image_org.cuda()
            predicted_score_org = model(image_org).detach().cpu().numpy()
            score_org_i.append(predicted_score_org)
            
            # test performance with no-compressed adv images
            img_adv_path = os.path.join(img_adv_folder+'/BMP',images_all[i][:-4] + '_' + str(j) + '.bmp')
            image_adv = pil_loader(img_adv_path)
            image_adv = transform_w_norm(image_adv)
            image_adv = image_adv.unsqueeze(0)
            image_adv = image_adv.cuda()
            predicted_score_adv_wo = model(image_adv).detach().cpu().numpy()
            score_adv_i_wo.append(predicted_score_adv_wo)
            
            # test performance with compressed adv images
            img_adv_path = os.path.join(img_adv_folder+'/JPEG70',images_all[i][:-4] + '_' + str(j) + '.jpeg')
            image_adv = pil_loader(img_adv_path)
            image_adv = transform_w_norm(image_adv)
            image_adv = image_adv.unsqueeze(0)
            image_adv = image_adv.cuda()
            predicted_score_adv_w = model(image_adv).detach().cpu().numpy()
            score_adv_i_w.append(predicted_score_adv_w)
            
        score_org_i = np.array(score_org_i)
        score_org_i = np.mean(score_org_i)
        org_pred.append(score_org_i)
        score_adv_i_wo = np.array(score_adv_i_wo)
        score_adv_i_wo = np.mean(score_adv_i_wo)
        adv_pred_wo.append(score_adv_i_wo)
        score_adv_i_w = np.array(score_adv_i_w)
        score_adv_i_w = np.mean(score_adv_i_w)
        adv_pred_w.append(score_adv_i_w)
    
    org_pred = np.array(org_pred)
    adv_pred_wo = np.array(adv_pred_wo)
    adv_pred_w = np.array(adv_pred_w)

    rho_s, _ = spearmanr(org_pred, adv_pred_wo)
    rho_p, _ = pearsonr(org_pred, adv_pred_wo)
    rho_k, _ = kendalltau(org_pred, adv_pred_wo)
    rmse = np.sqrt(np.mean(np.power((org_pred - adv_pred_wo),2)))
    mae =  np.mean(np.abs(org_pred - adv_pred_wo))
    r = r_robustness(org_pred, adv_pred_wo)
    print('The attack performance before compression is: \nRMSE/MAE/R/SROCC/PLCC/KROCC\n{0:.4f},{1:.4f},{2:.4f},{3:.4f},{4:.4f},{5:.4f}'.format(rmse,mae,r,rho_s,rho_p,rho_k))
    
    rho_s, _ = spearmanr(org_pred, adv_pred_w)
    rho_p, _ = pearsonr(org_pred, adv_pred_w)
    rho_k, _ = kendalltau(org_pred, adv_pred_w)
    rmse = np.sqrt(np.mean(np.power((org_pred - adv_pred_w),2)))
    mae =  np.mean(np.abs(org_pred - adv_pred_w))
    r = r_robustness(org_pred, adv_pred_w)
    print('The attack performance after compression is: \nRMSE/MAE/R/SROCC/PLCC/KROCC\n{0:.4f},{1:.4f},{2:.4f},{3:.4f},{4:.4f},{5:.4f}'.format(rmse,mae,r,rho_s,rho_p,rho_k))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nt_weight', dest='nt', type=float, default=0, help='the weight of the NT strategy')

    config = parser.parse_args()
    main(config)