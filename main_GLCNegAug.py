#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 13:42:16 2021

@author: user1
"""

import torch, sys
import os, csv
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from utils import (AverageMeter, Logger, Memory, ModelCheckpoint,
                   NoiseContrastiveEstimator, LocalTripletLoss,LocalNegContrastiveEstimator, Progbar)
from datasets.GLCDataset_NegAug import GLContrastLoader
from network import Network

device = torch.device('cuda:0')
# train_data_dir = '/home/user1/PhD_CAPSULEAI/Project2021/DATA/train_contrast/dummy/'
# val_data_dir = '/home/user1/PhD_CAPSULEAI/Project2021/DATA/train_down/dummy/'
train_data_dir = '/home/user1/PhD_CAPSULEAI/Project2021/DATA/train_contrast/train/'
train_knn = '/home/user1/PhD_CAPSULEAI/Project2021/DATA/train_down_giana/train/'
val_data_dir = '/home/user1/PhD_CAPSULEAI/Project2021/DATA/train_down_giana/test/'
negative_nb = 500 # number of negative examples in NCE
lr = 0.12 #.003 loss; 0.0271  #  253 .008 0.0312  #327 0.006 0.0233
checkpoint_dir = 'models/WINCON_500Negatives'
log_filename = 'logs/GLC_NA/glc3_na_log'
dataparallel = True
resume_epoch = 0
max_epochs = 800
only_test = False
augmentation = 'aggr' 
results = {'test_acc@1': {}}
color_space = 'RGB'  # can be 'RGB' OR 'LAB'



alpha, beta = 0.5, 0.5
dataset = GLContrastLoader(train_data_dir, aug=augmentation)
train_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=100, num_workers=32)
mem_path = 'repr/WINCON_512_representations.pt'
tb_dir ='./tbx/WINCON_500Negatives/'


# test using KNN monitor:
train_knn_dataset = GLContrastLoader(root_dir=train_knn, if_test=True)
train_knn_loader = torch.utils.data.DataLoader(train_knn_dataset,shuffle=True,batch_size=64, num_workers=16) 
val_dataset = GLContrastLoader(root_dir=val_data_dir, if_test=True)
val_loader = torch.utils.data.DataLoader(val_dataset,shuffle=True,batch_size=64, num_workers=16) 

checkpoint = ModelCheckpoint(mode='min', directory=checkpoint_dir)
net = Network()
if not dataparallel:
    net = net.to(device)
else:
    net = torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count()))).cuda()
    
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=12e-5) 


if resume_epoch!= 0:
    resume_epoch = checkpoint.retreive_model(net, optimizer, resume_epoch)


memory = Memory(size=len(dataset), weight=0.5, device=device, path= mem_path)
memory.initialize(net, train_loader, epoch=resume_epoch)



noise_contrastive_estimator = NoiseContrastiveEstimator(device)
# local_negative_contrastive_estimator = LocalNegContrastiveEstimator(device)

# local_triplet_loss = LocalTripletLoss(device)

logger = Logger(log_filename, tb_dir)
loss_weight = 0.5

# test using a knn monitor
def test(train_knn_loader, val_loader):
    net.eval()
    classes = len(train_knn_loader.dataset.classes)  # get classes during test 
    total_top1,  total_num, feature_bank, clases = 0.0, 0, [], []
    with torch.no_grad():
        # generate feature representations 
        for imgs, target in train_knn_loader:  #TODO
            images = imgs.to(device)
            feature = net(images=images, patches=None, mode=0)  
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            clases.append(target)
            
        # [D, N]
        feature_bank = torch.cat(feature_bank).t().contiguous()
        # [N]
        feature_labels = torch.cat(clases).to(device)
        # loop test data to predict the label by weighted knn search
        test_bar = val_loader
        targets, features  = [], [] 
        
        for data, target in test_bar:
            data, target = data.to(device), target.to(device)
            feature = net(images=data, patches=None, mode=0)
            feature = F.normalize(feature, dim=1)
            
            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, knn_k=290, knn_t=0.1)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            targets.append(target)
            features.append(feature)
    test_acc_1 = total_top1 / total_num * 100
    # results['test_acc@1'][epoch] = test_acc_1
    print('\n Unsupervised Accuracy : {}'.format(round(test_acc_1)))
    if only_test:
        sys.exit()
        print( 'ADDING EMBEDDING FOR TEST DATA' )
        
    # logger.embedding(torch.cat(features), torch.cat(targets), epoch)
    # sys.exit()
    # save statistics:
        
    if os.path.isfile('./logs/GLCV/wincon_512_KNN_log.csv'):
        with open(r'./logs/GLCV/wincon_512_KNN_log.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, round(test_acc_1)])

    else:
        with open('./logs/GLCV/wincon_512_KNN_log.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, round(test_acc_1)])
    return


# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels
        
for epoch in range(int(resume_epoch)+1, max_epochs):
    
    if only_test is True or (epoch%10 == 0 and epoch>0):
        torch.cuda.empty_cache()
        test(train_knn_loader, val_loader)
    else: 
        iters=len(train_loader)
        net = net.train()  
        memory.update_weighted_count(epoch)
        train_loss = AverageMeter('train_loss')
        local_loss_3 = AverageMeter('local_negative_loss')
        global_loss_2 = AverageMeter('global_negative_loss')
        alternate_view_loss_1 = AverageMeter('jigsaw_loss')
        
        bar = Progbar(len(train_loader), stateful_metrics=['train_loss', 'valid_loss'])
        # lr = optimizer.param_groups[0]['lr']
        lr= scheduler.get_lr()
        print('\nEpoch: {}\t lr:{:.3f}'.format(epoch, lr[0]))
        for step, batch in enumerate(train_loader):
            # prepare batch
            images = batch['original'].to(device)  #TODO
            local_negatives = batch['local_negative'].to(device) # [64, 1, 224,224, 3] # merge the local negatives of the batch to make 64 local negatives.  
            patches = [element.to(device) for element in batch['patches']] 
            index = batch['index']
            representations = memory.return_representations(index).to(device).detach()
            # zero grad
            optimizer.zero_grad()
    
            #forward, loss, backward, step
            output = net(images=images, patches=patches, mode=1)
            output_local_negatives = net(images=local_negatives, mode=0)
            loss_1 = noise_contrastive_estimator(output[0], output[1], index, memory, negative_nb=negative_nb, local_negatives = output_local_negatives)  # patches
            loss_2 = noise_contrastive_estimator(output[0], representations, index, memory, negative_nb=negative_nb, local_negatives = output_local_negatives)  # original
            # loss_3 = local_negative_contrastive_estimator(output[0],representations, output_local_negatives)                  # Local Negatives
            loss = alpha* loss_1 + beta * loss_2
            loss.mean().backward()
            optimizer.step()
            scheduler.step(epoch + step / iters)
            # update representation memory
            memory.update(index, output[0].detach().cpu().numpy())
            
    
            # update metric and bar
            train_loss.update(loss.item(), images.shape[0])
            # local_loss_3.update(loss_3.item(), images.shape[0])
            global_loss_2.update(loss_2.item(), images.shape[0])
            alternate_view_loss_1.update(loss_1.item(), images.shape[0])
               
            bar.update(step, values=[('train_loss', train_loss.return_avg())])
        lr = scheduler.get_lr()[0]
        # update annealed lr before warm restart :
        logger.update(epoch=epoch, loss=train_loss.return_avg(), lr=lr, name='_full_')
        
        # sending all losses to tbx logging:
        # logger.update(epoch=epoch, loss=local_loss_3.return_avg(), lr=lr, name='_LN_')
        logger.update(epoch=epoch, loss=global_loss_2.return_avg(), lr=lr, name='_GN_')
        logger.update(epoch=epoch, loss=alternate_view_loss_1.return_avg(), lr=lr, name='_inter_view_')
        
        # embedding to check where LN, GN and Positives are in embedding space
        LN_feat = output_local_negatives.to(device).detach() # 64,128
        Prior_feat = output[0].to(device).detach()  # 64,128
        Jigsaw_feat = output[1].to(device).detach()  # 64,128
        GN_feat =  memory.return_random(size = negative_nb, index = index)  # 200,128
        GN_feat = torch.Tensor(GN_feat).to(device).detach()
        concat_feat = torch.cat((LN_feat, Prior_feat, GN_feat, Jigsaw_feat), 0)
        concat_labels = ['LN']*LN_feat.shape[0] + ['V_p']*Prior_feat.shape[0] + ['GN']*GN_feat.shape[0] +  ['V_d']*Jigsaw_feat.shape[0]
        logger.embedding(concat_feat ,label_imgs=None, meta=concat_labels, epoch=epoch)

        # Reset LR for annealing 
        # print('Reset scheduler after each epoch')
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader))
    
    
        # save model if improved
        checkpoint.save_model(net, optimizer, train_loss.return_avg(), epoch, memory)
        
        
        

