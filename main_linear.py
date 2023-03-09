#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 13:11:42 2021

@author: user1
"""
import os, csv
from collections import Counter
import torch
import torch.optim as optim
import torch.nn as nn
from utils import (AverageMeter, Logger, accuracy, ModelCheckpoint,Progbar, get_lr, test_metrics)
from datasets.Dataset import linear_loader
from network import Network


# GLC_NeGaUG :299/398, PIRL :630, GCV2 : 801/458 PGCONV3:409 797, win : 674
# D_GIANA = '/home/user1/PhD_CAPSULEAI/Project2021/DATA/train_down_balanced/GIANA'
# D_KID = '/home/user1/PhD_CAPSULEAI/Project2021/DATA/train_down_balanced/KID'
# w_kid = torch.tensor([0.2, 0.2, 0.1, 0.2, 0.1, 0.1, 0.1])
# D_KID2 = '/home/user1/PhD_CAPSULEAI/Project2021/DATA/train_down_balanced/KID2'
# D_KID2 = '/home/user1/PhD_CAPSULEAI/Project2021/DATA/train_down_KID2/KID2'
# w_kid2 = torch.tensor([0.1, 0.2, 0.2, 0.1])
# D_KVASIR = '/home/user1/PhD_CAPSULEAI/Project2021/DATA/train_down_balanced/KVASIR'
# D_KVASIR = '/home/user1/PhD_CAPSULEAI/Project2021/DATA/train_down_KVASIR/KVASIR'


#Binary

D_GIANA = '/home/user1/PhD_CAPSULEAI/Project2021/DATA/Binary/GIANA'
D_KID2 = '/home/user1/PhD_CAPSULEAI/Project2021/DATA/Binary/KID2'
LIN_DIR = D_KID2
d_name = LIN_DIR.split('/')[-1]
device = torch.device('cuda:0')
lr = 10
# initial_checkpoints_dir = 'models/WIN_Con_models'
# initial_checkpoints_dir = 'GCV_Rand_Prior_models'
# initial_checkpoints_dir = 'models/PGCon_MLPHead_L2'
# initial_checkpoints_dir = 'vanilla_models'
# initial_checkpoints_dir = ''
# initial_checkpoints_dir = 'models/GCV2_anchorprior'  # flipped anchor 
initial_checkpoints_dir = 'models/PGCon_v3'  # flipped anchor + transformation (CJ) in jigsaw patches
# initial_checkpoints_dir = 'models/GLCT_models'
initial_epoch = 344 #341, 465, 536, 558
resume_epoch = 0
max_epochs = 250
dataparallel = True
train_batchnorm = False
head = 'linear'  # or 'mlp
LINEAR_EVAL = False  # or Full Fine tune if False
# log_filename = 'Lin_eval_log'
test_only = False
label_percent = 100
## Contrastive Options
# exp_name = str(initial_epoch)+'_Contv1_FFT_'+str(label_percent)
# dirc = os.path.join('tbx/Contrastive_v1/', exp_name)
# lin_checkpoint_dir = os.path.join('lin_models/', exp_name) 
# log_filename = exp_name
# folder_name = 'GLC_NegAug'
# folder_name = 'GLC_Triplet'
# folder_name = 'vanilla_PIRL'
# folder_name = 'PGCon_v2_flipped_anchor'
folder_name = 'PGCon_v3'
# folder_name = 'GCV_RP'
# folder_name = 'RandInit'
# folder_name = 'INET_pretrained'
# folder_name = 'PGCon_MLPHead_L2'
# folder_name = 'PGCon_MLPHead_L2'
warm_fft =  False
warm_epoch = 280
## PIRL options
eval_type = '_warm_FFT_' if warm_fft is True else '_LE_' if LINEAR_EVAL is True else  '_FFT_'
exp_name = str(initial_epoch)+'_'+folder_name+eval_type+str(label_percent)
dirc = os.path.join('tbx/Linear_Eval/binary/',folder_name,d_name, exp_name)
lin_checkpoint_dir = os.path.join('lin_models/binary/', folder_name, d_name, exp_name) 
log_filename = exp_name

Random_Init = False  # true to evaluate randomly initialized ResnET 50 
pretrained = False # set TRue only for Pretrained ImageNet evaluation
train_loader, classes = linear_loader(LIN_DIR, batch_size=128, train_val_test = 'train', label_percent=label_percent)
num_classes = len(classes)
print(classes)
val_loader, _ = linear_loader(LIN_DIR, batch_size=1204, train_val_test = 'val', label_percent=100)
test_loader, _ = linear_loader(LIN_DIR, batch_size=1204, train_val_test = 'val', label_percent=100)



# initate network
if not pretrained :
    net = Network(num_classes=num_classes)
else:
    net = Network(pretrained=pretrained, num_classes=num_classes)
    print('Imported ImageNet Wights to ResNet 50')
if not dataparallel:
    net = net.to(device)
else:
    net = torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count()))).cuda()
   
   
#init ckp to save and resume Linear training
lin_checkpoint = ModelCheckpoint(mode='min', directory=lin_checkpoint_dir)
if not Random_Init :
    if resume_epoch == 0:
        # Load Initial NCE Weights
        lin_checkpoint.retreive4linear(net, initial_epoch, initial_checkpoints_dir)
    else:
        # retrieve model
        pass
        print('Random INitialization')
        # lin_checkpoint.retreive_model(model, optimizer, epoch)
    
        
    ## Freeze Contrastive Layers that are NOT in use :
    for name, param in net.named_parameters():
        if 'projection_original_features' in name or'connect_patches_feature' in name:
            param.requires_grad = False 
    ########################################################################
    #######        LINEAR EVAL - Freeze all CONV layers               ######
    ########################################################################
    if LINEAR_EVAL or warm_fft:
        print('LINEAR EVALUATION ...')
        for param in net.module.network.parameters():
            param.requires_grad = False
        # for name, param in net.named_parameters():
        #     print(name, param.requires_grad)
            # unfreeze batchnorm scaling
            if train_batchnorm:
                for layer in net.modules():
                    if isinstance(layer, torch.nn.BatchNorm2d):
                        for param in layer.parameters():
                            param.requires_grad = True
        if head == 'linear':  
            for param in net.module.mlp_head.parameters():
                param.requires_grad = False
                # for name, param in net.named_parameters():
                #     print(name, param.requires_grad)
        elif head == 'mlp':
            for param in net.module.lin_head.parameters():
                param.requires_grad = False
    else:       
    ########################################################################
    #######        FULL FINE TUNING - train all CONV + Linear layers #######
    ########################################################################
        for name, param in net.named_parameters():
            print(name, param.requires_grad)
        print('FINE TUNING ...')
    
    # initialize optimizer and scheduler 
    # Verify params to be nn.Linear and optionally batchnorm
else:
    print('RANDOM INITIALIZATION ...')

optimizer = optim.SGD(filter(lambda x: x.requires_grad, net.parameters()), lr=lr, momentum=0.9, weight_decay=1e-4)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 3 , T_mult=2, eta_min=12e-5) 
logger = Logger(log_filename, dirc)
# loss
criterion = nn.CrossEntropyLoss().cuda()

# set model to eval to freeze batchnorma øayers
# net = net.eval()

def Val_Test(loader, val=True):
    val_loss = AverageMeter(train_val_test+'loss')
    top1 = AverageMeter(train_val_test+'top1')
    precision = AverageMeter(train_val_test+'precision')
    recall = AverageMeter(train_val_test+'recall')

    bar = Progbar(len(loader), stateful_metrics=['lin_valid_loss'])
    for step, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            output = net(images, patches=None, mode=2)
            loss = criterion(output, labels)
            prec1, pr, rc, fscore, support, c_mat, predp = test_metrics(output.cpu().data, labels.cpu().data, exp_name,  topk=(1,), val=False)

        # update metric and bar
        val_loss.update(loss.item(), images.shape[0])
        top1.update(prec1[0].item(), images.shape[0])
        precision.update(pr.item(),  images.shape[0])
        recall.update(rc.item(), images.shape[0])
        bar.update(step, values=[('lin_loss_'+ train_val_test, val_loss.return_avg())])
        lr = get_lr(optimizer)
    print('\n###############    VAL   ###############\n'
        # '\nEpoch: [{}][{}/{}]\t LR: {:.3f}\t'
              'Loss {loss.val:.3f} ({loss.avg:.3f})\n'
              'top1_acc {top1.val:.2f} ({top1.avg:.2f})\n'
              'precision {prec.val:.2f} ({prec.avg:.2f})\n'
              'recall {rec.val:.2f} ({rec.avg:.2f})\t'
              .format(loss=val_loss, top1=top1, prec=precision, rec=recall))
    print( '\n####################### Confusion matrix ###############\n')
    print(c_mat)
    print( '\n####################### Confusion matrix ###############\n')
    # result[fold] = {top1, pr, rc, fscore, support}
    logger.update(epoch, val_loss.return_avg(), lr, train_val_test, prec1, '_Lin_')
    if os.path.isfile('./logs/'+exp_name):
        with open(r'./logs/'+exp_name+'.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(prec1)

    else:
        with open('./logs/'+exp_name+'.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(prec1)
    




train_loss = AverageMeter('lin_train_loss')
top1 = AverageMeter('top_1')
iters =len(train_loader)

for epoch in range(int(resume_epoch), max_epochs):
    if warm_fft and epoch == warm_epoch:
        print('***************************************************************************')
        print('************************ Warm-up complete, training all layers ************')
        print('***************************************************************************')
        for param in net.module.network.parameters():
            param.requires_grad = True
        for name, param in net.named_parameters():
            print(name, param.requires_grad)
    if test_only:
        lin_checkpoint.retreive_model(net, optimizer, resume_epoch)
        net.eval()
        train_val_test = 'test'
        Val_Test(test_loader, val=False)
        import sys
        sys.exit()
    if epoch%5 == 0 and epoch != 0:
        net.eval()
        train_val_test = 'val'
        Val_Test(val_loader)
    net.train()
    # freeze batch norm layers from contrastive training - 
    train_val_test = 'train'
    # bar = Progbar(len(train_loader), stateful_metrics=['lin_train_loss'])
    for step, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        output = net(images, patches=None, mode=2)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        scheduler.step(epoch + step / iters)
        prec1, _ = accuracy(output.data, labels, topk=(1, 2))
        # update metric and bar
        train_loss.update(loss.item(), images.shape[0])
        top1.update(prec1[0], images.shape[0])
        # bar.update(step, values=[('lin_train_loss', train_loss.return_avg())])
        print('Epoch: [{}][{}/{}]\t'
                  'Loss {loss.val:.6f} ({loss.avg:.6f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  .format(epoch, step, len(train_loader), loss=train_loss, top1=top1))
        
    lr = get_lr(optimizer)
    logger.update(epoch, train_loss.return_avg(), lr, train_val_test, prec1, '_Lin_')

    # save model if improved
    lin_checkpoint.save_model(net, optimizer, train_loss.return_avg(), epoch, memory=None)
        
    if epoch == max_epochs-1:
        net.eval()
        train_val_test = 'test'
        Val_Test(test_loader, val=False)
        import sys
        sys.exit()
        
    

        
    
    
    