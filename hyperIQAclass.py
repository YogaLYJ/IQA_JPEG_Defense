# 为了方便直接调用，写进一个包里
from typing import Any
import torch
from torch import nn
import sys
# sys.path.append('/home/ycx/program/RobustIQA')
# from hyperIQA import models
import models
import os

class HyperIQA(nn.Module):
    def __init__(self,modelpath=None,LIVE=True):
        super(HyperIQA, self).__init__()
        self.model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
        self.model_hyper.train(False)
        if modelpath is not None:
            model_path = modelpath
            model_dict = torch.load(model_path)
            if 'model' in model_dict:
                self.model_hyper.load_state_dict((model_dict['model']))
            else:
                self.model_hyper.load_state_dict((model_dict))

    def forward(self,img): 
        paras = self.model_hyper(img)
        # Building target network
        model_target = models.TargetNet(paras).cuda()
        model_target.train(False)
        for param in model_target.parameters():
            param.requires_grad = False
        # Quality prediction
        pred = model_target(paras['target_in_vec'])
        pred = pred.clamp(0,100)
        
        return pred