#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 13:48:23 2021

@author: user1
"""


import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50
import torch.nn.functional as F

class Normalize(nn.Module):
    def __init__(self, p=2):
        super(Normalize, self).__init__()
        self.p = p

    def forward(self, x):
        return F.normalize(x, p=self.p, dim=1)

class Network(nn.Module):
    def __init__(self, pretrained=False, drop=False, num_classes=3):
        super(Network, self).__init__()
        self.drop = drop
        self.linear_size = 2048 if not self.drop else 1024
        self.network = resnet50(pretrained)
        self.network = torch.nn.Sequential(*list(self.network.children())[:-1])
        self.num_classes = num_classes
        if self.drop :
            # drop last conv layers
            self.network = nn.Sequential(*list(self.network.children())[:-2],list(self.network.children())[-1])
        else:
            self.network = self.network
            
        self.projection_original_features = nn.Linear(2048, 128)
        self.connect_patches_feature = nn.Linear(1152, 128)   # 128*9 = 1152
        self.lin_head = nn.Sequential(
                nn.Linear(self.linear_size, 128),
                Normalize(2))
        self.mlp_head = nn.Sequential(
                nn.Linear(self.linear_size, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 128),
                Normalize(2))
        self.classifier = nn.Linear(128, self.num_classes) 
        

    def forward_once(self, x):
        return self.network(x) #  [64,3,224,224] 
    

    def return_reduced_image_features(self, original):
        original_features = self.forward_once(original)
        original_features = original_features.view(-1, 2048)
        original_features = self.projection_original_features(original_features)
        return original_features                             # shape = (batch_size, 128)

    def return_reduced_image_patches_features(self, original, patches):
        original_features = self.return_reduced_image_features(original)

        patches_features = []
        for i, patch in enumerate(patches):
            patch_features = self.return_reduced_image_features(patch)
            patches_features.append(patch_features)

        patches_features = torch.cat(patches_features, axis=1) # [batch size, 128]
        patches_features = self.connect_patches_feature(patches_features)
        return original_features, patches_features
    
    def return_downstream_prob(self, x,  head='linear'):
        encoder_features = self.forward_once(x)   # [64,2048,1,1] dropped = [64, 1024, 1,1]
        encoder_features = encoder_features.view(-1,self.linear_size)  # [64,2048] or # [64,1024]

        if head == 'linear':
            feat = self.lin_head(encoder_features)
        else:
            feat = self.mlp_head(encoder_features)
        return self.classifier(feat)

    def forward(self, images=None, patches=None,  mode=0):
        '''
        mode 0: get 128 feature for image, 
        mode 1: get 128 feature for image and patches    
        mode 2: Linear or mlp head for eval 
        mode 3: semantic segmentation, pass through FCN head  # not incoporated yet
        '''
        
        if mode == 0:               
            return self.return_reduced_image_features(images)
        if mode == 1:
            return self.return_reduced_image_patches_features(images, patches)
        if mode == 2:
            return self.return_downstream_prob(x=images)  # Linear Eval without dropping conv layers
        # if mode == 3:

class RegLog(nn.Module):
    """Creates logistic regression on top of frozen features"""
    def __init__(self, conv, num_labels):
        super(RegLog, self).__init__()
        self.conv = conv
        if conv==1:
            self.av_pool = nn.AvgPool2d(6, stride=6, padding=3)
            s = 9600
        elif conv==2:
            self.av_pool = nn.AvgPool2d(4, stride=4, padding=0)
            s = 9216
        elif conv==3:
            self.av_pool = nn.AvgPool2d(3, stride=3, padding=1)
            s = 9600
        elif conv==4:
            self.av_pool = nn.AvgPool2d(3, stride=3, padding=1)
            s = 9600
        elif conv==5:
            self.av_pool = nn.AvgPool2d(2, stride=2, padding=0)
            s = 9216
        self.linear = nn.Linear(s, num_labels)

    def forward(self, x):
        x = self.av_pool(x)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        return self.linear(x)


    def forward(x, model, conv):
        if hasattr(model, 'sobel') and model.sobel is not None:
            x = model.sobel(x)
        count = 1
        for m in model.features.modules():
            if not isinstance(m, nn.Sequential):
                x = m(x)
            if isinstance(m, nn.ReLU):
                if count == conv:
                    return x
                count = count + 1
        return x
