#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 12:26:59 2021

@author: Anuja
"""

'''KID '''
import random, os, sys

import numpy as np
from skimage import color
from collections import Counter
import torch
import torchvision
from torchvision.datasets import DatasetFolder
from torch.utils.data import ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold as SKFold


sys.path.append('/home/user1/PhD_CAPSULEAI/Project2021/src/')
from utils import pil_loader

extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
mean = [0.56, 0.35, 0.20]       
std = [0.3, 0.24, 0.17]
neighbors = {
        1: [2,5,6,7,8],
        3: [0,3,6,7,8],
        2: [0,1,2,5,8],
        4: [0,1,2,3,6],
        0: [0,1,2,3,5,6,7,8]
        }
class RGB2LAB(object):
    ''' Taken from InfoMin paper'''
    """Convert RGB PIL image to ndarray Lab."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2lab(img)
        return img
    
class RGB2YCbCr(object):
    """Convert RGB PIL image to ndarray YCbCr."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2ycbcr(img)
        return img


class RGB2YDbDr(object):
    """Convert RGB PIL image to ndarray YDbDr."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2ydbdr(img)
        return img


class RGB2YPbPr(object):
    """Convert RGB PIL image to ndarray YPbPr."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2ypbpr(img)
        return img
   
class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor  

def correct_overflow(xp, yp, bsize, side):
    if xp-bsize<0:
        xp=bsize
    if yp-bsize<0:
        yp=bsize
    if xp+bsize>side:
        xp -= xp+bsize-side
    if yp+bsize>side:
        yp -= yp+bsize-side
    return xp,yp
    
class ContrastLoader(DatasetFolder):
    def __init__(self, root_dir, if_test=False):
        super(ContrastLoader, self).__init__(root_dir, pil_loader, extensions=extensions)
        self.if_test = if_test
        self.rgb2lab = RGB2LAB()
        self.root_dir = root_dir
        self.color_transform_prior = torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0)  
        self.color_transform_mod = torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0) 
        self.flips = [torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomVerticalFlip()]
        self.normalize = torchvision.transforms.Normalize(mean, std)
        self.grayscale = torchvision.transforms.RandomGrayscale(p=0.1)
        self.erase = torchvision.transforms.RandomErasing(p=0.5,scale=(0.02, 0.3), ratio=(0.3,2))
        self.inv_transform = torchvision.transforms.Compose([Denormalize(mean, std), 
                                                            lambda x: x.numpy()*255.0,  
                                                            lambda x: x.transpose(1,2,0).astype(np.uint8)
                                                            ])
        

    def pre_transform(self, img):
        size = img.size[0]
        margin=50
        # check size and centre-crop biggest central patch, assign bbox size 
        if size<500:  # KID (360,360) and OSF (336, 336)
            side, bsize = 316, 50
            jigsaw_size = 280 if size < 360 else 300 
        else:    # Sheffield (576x576)
            side, bsize=500, 75
            jigsaw_size = 400
        prior_patch = torchvision.transforms.CenterCrop((side,side))(img)
        a_channel = self.rgb2lab(prior_patch)[:,:,1] # a channel 
        #Random prior
        # x_p, y_p = np.unravel_index(a_channel.argmax(), a_channel.shape)
        # Selected prior 
        y_p, x_p = np.unravel_index(a_channel.argmax(), a_channel.shape)
        
        if x_p<(side//2):
            quad = 1 if y_p<(side//2) else 3
        else:
            quad = 2 if y_p<(side//2) else 4
        if ((side//2-margin <x_p< side//2+margin) and (side//2-margin <y_p< side//2+margin)):
            quad = 0
        else:
            quad
        return prior_patch, bsize, jigsaw_size, (x_p, y_p), quad 
        
        
    def __getitem__(self, index):
        """
         Args:
             index : index of images in batch
         return:
             dict
             original : 200x200 patch with prior suspect
             patches  : 9 jumbled patches with 64x64x3 for jigsaw, permutation index not returned as this is unsupervised
             index : index of sample for memory update 
        """
        if not self.if_test:
            path, _ = self.samples[index]
            original = self.loader(path)
            # Prior suspect patch extraction 
            prior_patch, bsize, jigsaw_size, (x_p, y_p), quadrant = self.pre_transform(original) # trimmed original
            # check margin and shift centre if needed
            xp, yp = correct_overflow(x_p,y_p, bsize, prior_patch.size[0])
            prior_suspect = prior_patch.crop((xp-bsize, yp-bsize, xp+bsize, yp+bsize))     #(left, top, right, bottom)  
            # optional plots
            # plt.imshow(original)
            # plt.show()
            # plt.imshow(prior_patch)
            # plt.scatter(xp,yp, color = 'red')
            # plt.show()
            
            # the size of prior suspect is either 100 or 200, resize to 224 for input to Resnet-50
            prior_image = torchvision.transforms.Resize((120, 120))(prior_suspect)  # 300/200 --> 224
            # add standard transforms  (KEEP PATHOLOGY PRISTINE)
            # prior suspect, do not change hue
            image=self.color_transform_prior(prior_image)
            # augmentation - flips
            image = self.flips[0](prior_image)
            image = self.flips[1](image)
            # To tensor
            image = torchvision.transforms.functional.to_tensor(image)
            # normalize
            image = self.normalize(image)
            
            # Jigsaw tranformation 
            # Crop original to remove corner pixels
            sample = torchvision.transforms.CenterCrop((jigsaw_size))(original)    # corners removed
            # sample = torchvision.transforms.Resize((255,255))(sample)  # resize jigsaw image to (255,255) 400/300/280 --> 255
            # crop into patches 
            pw = sample.size[0] // 3   # pw = 133/93/100
            crop_areas = [(i*pw, j*pw, (i+1)*pw, (j+1)*pw) for i in range(3) for j in range(3)]
            samples = [sample.crop(crop_area) for crop_area in crop_areas]
            samples = [torchvision.transforms.RandomCrop((90, 90))(patch) for patch in samples]  #TODO workout size to crop
            # In most cases, pathologies can be present in max 4 patches, distort randomly other remaining patches by random color-trasnforms and erasing.
            patches2distort = neighbors[quadrant]
            # augmentation color jitter (all patches for trasnform invarinace, otherwise only trasnform mod)
            # samples = [self.color_transform_mod(patch) if pid in patches2distort else patch for pid, patch in enumerate(samples)]
            samples = [self.color_transform_mod(patch) for patch in samples]
            # grayscale
            samples = [self.grayscale(patch) if pid in patches2distort else patch for pid, patch in enumerate(samples)]
            # to tensor
            samples = [torchvision.transforms.functional.to_tensor(patch) for patch in samples]
            # normalize
            samples = [self.normalize(patch) for patch in samples]
            # erase randomly
            samples = [self.erase(patch) if pid in patches2distort else patch for pid, patch in enumerate(samples)] 
            random.shuffle(samples)
            return {'original': image, 'patches': samples, 'index': index}

        else:
             path, target = self.samples[index]
             original = self.loader(path)
             crop_size = 280 if original.size[0]<500 else 400
             # remove corners
             sample = torchvision.transforms.CenterCrop((crop_size))(original) 
             sample= torchvision.transforms.Resize((224, 224))(sample)
             sample = torchvision.transforms.functional.to_tensor(sample)
             sample = self.normalize(sample)
             return sample, target
            

def linear_loader(root_dir, batch_size, train_val_test, label_percent):
    transform = { 'train' : torchvision.transforms.Compose([torchvision.transforms.CenterCrop((410)),  
                                                                torchvision.transforms.Resize((224)),
                                                                torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.6, saturation=0, hue=0),
                                                                torchvision.transforms.RandomHorizontalFlip(),
                                                                torchvision.transforms.RandomVerticalFlip(),
                                                                torchvision.transforms.ToTensor(),
                                                                torchvision.transforms.Normalize(mean, std)
                                                         ]),
                          'test' : torchvision.transforms.Compose([torchvision.transforms.CenterCrop((410)),
                                                                   torchvision.transforms.Resize((224)),
                                                                   torchvision.transforms.ToTensor(),
                                                                   torchvision.transforms.Normalize(mean, std)
                                                         ]),
                          'val' : torchvision.transforms.Compose([torchvision.transforms.CenterCrop((410)),
                                                                   torchvision.transforms.Resize((224)),
                                                                   torchvision.transforms.ToTensor(),
                                                                   torchvision.transforms.Normalize(mean, std)])
                }
    root = os.path.join(root_dir, train_val_test)
    lin_dataset = torchvision.datasets.DatasetFolder(root=root, loader=pil_loader,transform=transform[train_val_test],extensions=extensions)
    train_indices = np.random.choice(range(len(lin_dataset)), (label_percent*len(lin_dataset) // 100), replace=False)
    classes = lin_dataset.classes
    if train_val_test == 'train':
        sampler = SubsetRandomSampler(train_indices)
    else:
        sampler = None
    lin_loader = torch.utils.data.DataLoader(dataset=lin_dataset, batch_size=batch_size, sampler=sampler, num_workers=8)
    chosen_targets = [lin_dataset.targets[i] for i in train_indices]
    print(Counter(chosen_targets))
    return lin_loader, classes
       
def KFold_loader(root_dir, label_percent):
    
    transform = { 'train' : torchvision.transforms.Compose([torchvision.transforms.CenterCrop((410)),  
                                                                torchvision.transforms.Resize((224)),
                                                                torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.6, saturation=0, hue=0),
                                                                torchvision.transforms.RandomHorizontalFlip(),
                                                                torchvision.transforms.RandomVerticalFlip(),
                                                                torchvision.transforms.ToTensor(),
                                                                torchvision.transforms.Normalize(mean, std)
                                                         ]),
                          'test' : torchvision.transforms.Compose([torchvision.transforms.CenterCrop((410)),
                                                                   torchvision.transforms.Resize((224)),
                                                                   torchvision.transforms.ToTensor(),
                                                                   torchvision.transforms.Normalize(mean, std)
                                                         ]),
                          'val' : torchvision.transforms.Compose([torchvision.transforms.CenterCrop((410)),
                                                                   torchvision.transforms.Resize((224)),
                                                                   torchvision.transforms.ToTensor(),
                                                                   torchvision.transforms.Normalize(mean, std)])
                }
    
    labels=[]
    root_train = os.path.join(root_dir, 'train')
    train_dataset = torchvision.datasets.DatasetFolder(root=root_train, loader=pil_loader,transform=transform['train'],extensions=extensions)
    root_test = os.path.join(root_dir, 'test')
    test_dataset = torchvision.datasets.DatasetFolder(root=root_test, loader=pil_loader,transform=transform['test'],extensions=extensions)
    dataset = ConcatDataset([train_dataset, test_dataset])
    labels = train_dataset.targets + test_dataset.targets

    # Define the K-fold Cross Validator
    skfold = SKFold(n_splits=5, shuffle=True)
    return skfold, dataset, labels
    
    
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    DATA_DIR = '/home/user1/PhD_CAPSULEAI/Project2021/DATA/train_contrast/train/'
    LIN_DIR = '/home/user1/PhD_CAPSULEAI/Project2021/DATA/train_down/'
    contrast = True
    contrast_dataset = ContrastLoader(DATA_DIR)  # Contrastive training using KID, OSF and Sheffield data
    inv_transform = contrast_dataset.inv_transform
    if contrast : 
        train_loader =  torch.utils.data.DataLoader(contrast_dataset, batch_size=1, shuffle=True, num_workers=0)
        for sample in train_loader:
            prior_suspect, jigsaw_patches, index = [sample[key] for key in sample.keys()]
        
            ## Plot the images for sanity check
            ## plt prior suspect 
            
            if prior_suspect.shape[0] > 1:
                for i in range(prior_suspect.shape[0]):
                    plt.subplot((prior_suspect.shape[0]//4),4,i+1)
                    fig=plt.imshow(inv_transform(prior_suspect[i]))
                    fig.axes.get_xaxis().set_visible(False)
                    fig.axes.get_yaxis().set_visible(False)
            else:
                plt.imshow(inv_transform(prior_suspect[0]))
                plt.xticks([])
                plt.yticks([])
            plt.show()
            name='/home/user1/PhD_CAPSULEAI/Project2021/preprocessing/paper_images/preprep/' + str(index[0])+'.png'
            # plt.savefig(name)
                
                
            
            # plot jigsaw patches : list of 9x[Batch,64x64x3]
            pats= [jigsaw_patches[i][0] for i in range(len(jigsaw_patches))]
            patches = torch.stack(pats)
            patch_list = []
            for i in range(patches.shape[0]):
                patch_list.append(Denormalize(mean, std)(patches[i]))
            transformed = torch.stack(patch_list)
                
            
            # patches = torchvision.transforms.functional.to_tensor(transformed)
            grid_img = torchvision.utils.make_grid(transformed,3)
            plt.imshow(grid_img.permute(1, 2, 0))
            plt.xticks([])
            plt.yticks([])
            plt.show()
            name='/home/user1/PhD_CAPSULEAI/Project2021/preprocessing/paper_images/preprep/j_' + str(index[0])+'.png'
            # plt.savefig(name)
    else:
        transform = { 'train' : torchvision.transforms.Compose([torchvision.transforms.CenterCrop((410)),  #TODO size crop for Giana
                                                                torchvision.transforms.Resize((224)),
                                                                torchvision.transforms.RandomHorizontalFlip(),
                                                                torchvision.transforms.RandomVerticalFlip(),
                                                                torchvision.transforms.ToTensor(),
                                                                torchvision.transforms.Normalize(mean, std)
                                                         ]),
                          'test' : torchvision.transforms.Compose([torchvision.transforms.CenterCrop((410)),
                                                                   torchvision.transforms.Resize((224)),
                                                                   torchvision.transforms.ToTensor(),
                                                                   torchvision.transforms.Normalize(mean, std)
                                                         ])
                }
        train_val_test = 'train'
        root = os.path.join(LIN_DIR, train_val_test)
        inv_transform = contrast_dataset.inv_transform
        lin_dataset = torchvision.datasets.DatasetFolder(root=root, loader=pil_loader,transform=transform[train_val_test],extensions=extensions)
        classes = lin_dataset.classes
        lin_loader = torch.utils.data.DataLoader(dataset=lin_dataset, batch_size=4, shuffle=True)
        
        for step, batch in enumerate(lin_loader):
            images, labels = batch
            for i in range(images.shape[0]):
                plt.subplot((images.shape[0]//4), 4, i+1)
                fig=plt.imshow(inv_transform(images[i]))
                plt.title(classes[labels[i]])
                fig.axes.get_xaxis().set_visible(False)
                fig.axes.get_yaxis().set_visible(False)
            plt.show()
            if step >3:
                import sys
                sys.exit()
                

            
         
        
        
    
            
        
        
    
    
  
    
    
    