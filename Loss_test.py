#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 09:56:19 2021

@author: user1
"""


# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 15:54:39 2021

@author: anjuv
"""
import numpy as np
import torch, sys
import torch.optim as optim

import matplotlib.pyplot as plt

sys.path.append('/home/user1/PhD_CAPSULEAI/Project2021/src/')
torch.manual_seed(0)
np.random.seed(0)

from utils import (Memory, ModelCheckpoint)
from datasets.Dataset import ContrastLoader
from network import Network

device = torch.device('cuda:0')
# train_data_dir = '/home/user1/PhD_CAPSULEAI/Project2021/DATA/train_down/dummy/'
# val_data_dir = '/home/user1/PhD_CAPSULEAI/Project2021/DATA/train_down/dummy/'
train_data_dir = '/home/user1/PhD_CAPSULEAI/Project2021/DATA/train_contrast/train/'
negative_nb = 200 # number of negative examples in NCE
lr = 0.012
checkpoint_dir = 'models'
dataparallel = True
resume_epoch = 341
max_epochs = 700

device = torch.device('cuda:0')
dataset = ContrastLoader(train_data_dir)
train_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=32, num_workers=1)

checkpoint = ModelCheckpoint(mode='min', directory=checkpoint_dir)
net = Network()
if not dataparallel:
    net = net.to(device)
else:
    net = torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count()))).cuda()
    
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)


if resume_epoch!= 0:
    resume_epoch = checkpoint.retreive_model(net, optimizer, resume_epoch)


memory = Memory(size=len(dataset), weight=0.5, device=device)
memory.initialize(net, train_loader)
loss_weight = 0.5
temp = 0.07

locally_aware_NCEL = []
all_negatives_equal_InfoNCEL = []
Info_NCEL = []
locally_aware_INFONCEL =[]

local = []
_pseudo_LNL = []
class NoiseContrastiveEstimator():
    def __init__(self, device):
        self.device = device
    
    def __call__(self, R_orig, prior_positive, index, memory, local_negative):   
        loss = 0
        for i in range(R_orig.shape[0]): 
            temp = 0.07 
            cos = torch.nn.CosineSimilarity()
            criterion = torch.nn.CrossEntropyLoss()
            
            sim_positive = cos(R_orig[None, i,:], prior_positive[None, i,:])/temp
            
            glob_negative = memory.return_random(size = 100, index = [index[i]])
            glob_negative = torch.Tensor(glob_negative).to(self.device)
            if not local_negative:
                sim_negative = cos(prior_positive[None, i,:], glob_negative) / temp 
                similarities = torch.cat((sim_positive, sim_negative))
                loss += criterion(similarities[None,:], torch.tensor([0]).to(self.device))
            
            else:
                _pseudo_LN = torch.tensor(memory.return_random(size = 1, index = [index[i]])).to(self.device)
                _pseudo_LNL.append(_pseudo_LN)
                all_negtives = torch.cat((glob_negative, _pseudo_LN.float()))
                # all_negtives = torch.Tensor(all_negtives).to(self.device)
                sim_negative =  cos(prior_positive[None, i,:], all_negtives) / temp
                similarities = torch.cat((sim_positive, sim_negative))
                loss += criterion(similarities[None,:], torch.tensor([0]).to(self.device))
        if not local_negative:
            return loss / R_orig.shape[0] 
        else:
            return loss / R_orig.shape[0], _pseudo_LNL

                

class LocalContrastiveEstimator():
    def __init__(self, device):
        self.device = device 
        
    def __call__(self, R_orig, positive, _pseudo_LNL):
        loss = 0
        _pseudo_LNL = torch.stack(_pseudo_LNL).squeeze(1).to(self.device).detach()
        
        for i in range(R_orig.shape[0]):
            temp = 0.07
            cos = torch.nn.CosineSimilarity()
            criterion = torch.nn.CrossEntropyLoss()
            
            sim_positive = cos(R_orig[None, i, :], positive[None, i, :]) / temp
            sim_LN = cos(positive[None, i, :], _pseudo_LNL[None, i, :])/ temp
            similarities = torch.cat((sim_positive, sim_LN.float()))
            loss += criterion(similarities[None, :], torch.tensor([0]).to(self.device))
        return loss/R_orig.size(0)
            
               
iters=len(train_loader)
net = net.train()  
train_loss = []
noise_contrastive_estimator = NoiseContrastiveEstimator(device)
local_contrastive_estimator = LocalContrastiveEstimator(device)

for step, batch in enumerate(train_loader):
    # prepare batch
    with torch.no_grad():
        images = batch['original'].to(device)
        patches = [element.to(device) for element in batch['patches']]
        index = batch['index']
        R_orig = memory.return_representations(index).to(device).detach()  # Return batch representations by selected indices
    
        #forward, loss, backward, step
        output = net(images=images, patches=patches, mode=1)
        Z_J = output[1]
        Z_P = output[0]
        
        all_negatives_equal_InfoNCE, _pseudo_LN = noise_contrastive_estimator(R_orig, Z_P, index, memory, local_negative=True) # adds a pseudo negative acting as local negative
        Info_NCE = noise_contrastive_estimator(R_orig, Z_P, index, memory, local_negative=False)
        
        locally_aware_NCE = local_contrastive_estimator(R_orig, Z_P, _pseudo_LN)  # patches
        locally_aware_INFONCE = Info_NCE +   locally_aware_NCE
        
        all_negatives_equal_InfoNCEL.append(all_negatives_equal_InfoNCE)
        locally_aware_NCEL.append(locally_aware_NCE)
        Info_NCEL.append(Info_NCE)
        locally_aware_INFONCEL.append(locally_aware_INFONCE)
        # all_negatives_equal_InfoNCE = all_negatives_equal_InfoNCE.cpu().detach().numpy()
        # locally_aware_NCE.cpu().detach().numpy()
        # Info_NCE.cpu().detach().numpy()
        
    if step>10:
        plt.plot(range(len(all_negatives_equal_InfoNCEL)), all_negatives_equal_InfoNCEL, 'g', label= 'locally aware, equal wt to global+local ')
        plt.plot(range(len(locally_aware_NCEL)), locally_aware_NCEL, 'r', label='Local NCE ')
        plt.plot(range(len(Info_NCEL)), Info_NCEL, 'c', label='InfoNCE, locally unaware')
        plt.plot(range(len(locally_aware_INFONCEL)), locally_aware_INFONCEL, 'K', label='LOCally Aware InfoNCE')
        plt.legend()
        plt.show()
        sys.exit()

    

    
            
        
        # f1 = np.array([[1,1,1],[1,2,1]]).reshape((2,3))
        # f2 = np.array([[1,1,1]])
        
        # xcorr = signal.correlate2d(f1,f2)
        # print(xcorr)
        
