import collections, copy
import pandas as pd
import time, sys, csv
import numpy as np
import torch
import random
import os
import shutil
from PIL import Image
import matplotlib as mpl
import torch.nn.functional as F
import datetime
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_fscore_support
import matplotlib.pyplot as plt
from scipy import interp
from tensorboardX import SummaryWriter

mean = [0.56, 0.35, 0.20]       
std = [0.3, 0.24, 0.17]
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
 

 
def test_metrics(output, target, name, topk=(1,), val=True):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    cm = confusion_matrix(pred.squeeze(0), target)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    print(pred)
    print(target)
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    p,r,f,s = precision_recall_fscore_support(pred.squeeze(0), target, average='weighted', labels=np.unique(pred.squeeze(0))) # return fscore also during val
    conf_mat = np.array(cm)
    if val:
        return res, p, r, f, s, conf_mat, pred
    else:
        print_roc_curve(output.numpy(), target, name)
        return res, p, r, f, s,conf_mat, pred

class AverageMeter(object):
    '''
    Taken from:
    https://github.com/keras-team/keras
    '''
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    def return_avg(self):
        return self.avg
        
class Progbar(object):
    '''
    Taken from:
    https://github.com/keras-team/keras
    '''
    """Displays a progress bar.
    # Arguments
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1, interval=0.05,
                 stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        self._values = collections.OrderedDict()
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                # Stateful metrics output a numeric value.  This representation
                # means "take an average from a single value" but keeps the
                # numeric formatting.
                self._values[k] = [v, 1]
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = ('%d:%02d:%02d' %
                                  (eta // 3600, (eta % 3600) // 60, eta % 60))
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values:
                    info += ' - %s:' % k
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)


class Memory(object):
    def __init__(self, device, size=2000, weight = 0.5, path=None): #chnage to len dataset
        self.memory = np.zeros((size, 128))
        self.weighted_sum = np.zeros((size, 128))
        self.weighted_count = 0
        self.weight = weight
        self.device = device
        self.epoch = 0
        self.path = 'repr/representations.pt' if path == None else path
        self.running_memory_state = {}
        
    def initialize(self, net, train_loader, epoch):
        if os.path.isfile(self.path):
            print('Retreiving saved representations to memory')
            self.running_memory_state  = torch.load(self.path)
            self.memory = self.running_memory_state[epoch]['memory']
            self.weighted_sum = self.running_memory_state[epoch]['weighted_sum']
        else:       
            self.update_weighted_count(epoch)
            print('Saving representations to memory')
            bar = Progbar(len(train_loader), stateful_metrics=[])
            for step, batch in enumerate(train_loader):
                with torch.no_grad():                
                    images = batch['original'].to(self.device)
                    index = batch['index']
                    output = net(images = images, mode = 0)                
                    self.weighted_sum[index, :] = output.cpu().numpy()
                    self.memory[index, :] = self.weighted_sum[index, :]
                    bar.update(step, values= [])
            memory_state = {
                'memory' : self.memory,
                'weighted_sum' : self.weighted_sum}
            # torch.save(memory_state, self.path)
            self.running_memory_state[self.epoch] = memory_state
            torch.save(self.running_memory_state, self.path)
    
    def initialize_wout_ckp(self, net, train_loader, epoch):
        if os.path.isfile(self.path):
            print('Retreiving saved representations to memory')
            self.running_memory_state  = torch.load(self.path)
            self.memory = self.running_memory_state['memory']
            self.weighted_sum = self.running_memory_state['weighted_sum']
        else:       
            self.update_weighted_count(epoch)
            print('Saving representations to memory')
            bar = Progbar(len(train_loader), stateful_metrics=[])
            for step, batch in enumerate(train_loader):
                with torch.no_grad():                
                    images = batch['original'].to(self.device)
                    index = batch['index']
                    output = net(images = images, mode = 0)                
                    self.weighted_sum[index, :] = output.cpu().numpy()
                    self.memory[index, :] = self.weighted_sum[index, :]
                    bar.update(step, values= [])
            memory_state = {
                'memory' : self.memory,
                'weighted_sum' : self.weighted_sum}
            # torch.save(memory_state, self.path)
            self.running_memory_state[self.epoch] = memory_state
            torch.save(self.running_memory_state, self.path)
                
    def update(self, index, values, save_updated_reps=False):
        self.weighted_sum[index, :] = values + (1 - self.weight) * self.weighted_sum[index, :] 
        self.memory[index, :] = self.weighted_sum[index, :]/self.weighted_count
        pass
    def save_updated_reps(self, epoch):
        self.epoch = epoch
        memory_state = {
            'memory' : self.memory,
            'weighted_sum' : self.weighted_sum}
        self.running_memory_state[self.epoch] = memory_state
        torch.save(self.running_memory_state, self.path)
        print('Updates representations saved ')
    
    def update_weighted_count(self, epoch):
        self.weighted_count = 1 + (1 - self.weight) * self.weighted_count
        self.epoch = epoch
        

    def return_random(self, size, index):
        if isinstance(index, torch.Tensor):
            index = index.tolist()
        #allowed = [x for x in range(2000) if x not in index]
        allowed = [x for x in range(index[0])] + [x for x in range(index[0] + 1, 91190)] #TODO 91190 91190
        index = random.sample(allowed, size)
        return self.memory[index,:]
    def return_representations(self, index):
        if isinstance(index, torch.Tensor):
            index = index.tolist()
        return torch.Tensor(self.memory[index,:])

class ModelCheckpoint():
    def __init__(self, mode, directory):
        self.directory = directory
        if mode =='min':
            self.best = np.inf
            self.monitor_op = np.less
        elif mode == 'max':
            self.best = 0
            self.monitor_op = np.greater
        else:
            print('\nChose mode \'min\' or \'max\'')
            raise Exception('Mode should be either min or max')
        if not os.path.isdir(self.directory):
            # shutil.rmtree(self.directory)
            os.mkdir(self.directory)
        else: pass
    
    def load_unparallel(self, state_dict):
        # check if the keys are already compatible with data parallel, i.e, have prefix 'module'
        unparallel_dict = copy.deepcopy(state_dict)
        for key in state_dict.keys():
            if 'network' in key:
                new_key = key[15:]
                unparallel_dict[new_key] = unparallel_dict.pop(key)
            else:
                print('already un-parallel')
                break
                    
        return unparallel_dict
    

    def save_model(self, model, optimizer, current_value, epoch, memory):
        print(' \n*************** ======= Saving Model ======= ******************** \n')
        if self.monitor_op(current_value, self.best):
            print('\nSave model, best value {:.3f}, epoch: {}'.format(current_value, epoch))
            self.best = current_value
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
                }
            torch.save(state, os.path.join(self.directory,'epoch_{}'.format(epoch)))
            if memory is not None:
                # save a dict of memory for eah ckp, if None, only latest memory is saved instead of at every good ckp.
                memory.save_updated_reps(epoch)
            del state
                       
    def retreive_model(self, model, optimizer, epoch):
        state = torch.load(os.path.join(self.directory,'epoch_{}'.format(epoch)))
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        return epoch
    
    def retreive4linear(self, model, epoch, direc, drop=False):
        # retrieve contrastive checkpoint for linear evaluation 
        state = torch.load(os.path.join(direc,'epoch_{}'.format(epoch)))
        print( 'Contrastive checkpoint retrieved : '+direc+str(epoch))
        if not drop:
            # return all params from contrastive learning
            dropped_state = {k:v for k,v in state['model'].items() if 'classifier' not in k}
            model.load_state_dict(dropped_state, strict=False)
            # model.load_state_dict(state['model'])
        else:
            # Bottleneck 7 dropped 
            dropped_state = {k:v for k,v in state['model'].items() if '7' not in k and 'lin' not in k and 'mlp' not in k and 'classifier' not in k}
            model.load_state_dict(dropped_state, strict=False)
        # print(state['model'])
        # no need to retrieve optim params
        return 
    def retreive4segmentation(self, model, epoch, direc):
        state = torch.load(os.path.join(direc,'epoch_{}'.format(epoch)))
        state_unparallel = self.load_unparallel(state['model'])
        print( 'Contrastive checkpoint retrieved : '+direc+str(epoch))
        # return all params from contrastive learning
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in state_unparallel.items() if k in model.state_dict()}
        # 2. overwrite entries in the existing state dict
        model.state_dict().update(pretrained_dict) 
        # 3. load the new state dict
        model.load_state_dict(pretrained_dict)
        # print(state['model'])
        # no need to retrieve optim params
        return 
        
class NoiseContrastiveEstimator():
    def __init__(self, device):
        self.device = device
    
    def __call__(self, original_features, path_features, index, memory, negative_nb = 1000, local_negatives=None):   
        loss = 0
        # dummy_loss = []
        for i in range(original_features.shape[0]): 
            
            temp = 0.07 
            cos = torch.nn.CosineSimilarity()
            criterion = torch.nn.CrossEntropyLoss()
             
            negative = memory.return_random(size = negative_nb, index = [index[i]])
            negative = torch.Tensor(negative).to(self.device).detach()
            image_to_modification_similarity = cos(original_features[None, i,:], path_features[None, i,:])/temp  # cos(prior, jigasw) # cos(prior, rep_i) 
            if local_negatives is None:
                matrix_of_similarity = cos(original_features[None, i,:], negative) / temp   # cos (prior, negative)
            else:
                negative = torch.cat((negative,local_negatives)) # 264
                matrix_of_similarity = cos(original_features[None, i,:], negative) / temp   # cos (prior, negative)
            
            similarities = torch.cat((image_to_modification_similarity, matrix_of_similarity)) 
            # print(similarities.shape) 
            # dummy_loss.append((criterion(similarities[None,:], torch.tensor([0]).to(self.device))).item())
            loss += criterion(similarities[None,:], torch.tensor([0]).to(self.device))
    
            return loss / original_features.shape[0]
    
class LocalNegContrastiveEstimator():
    def __init__(self, device):
        self.device = device 
        
    def __call__(self, prior_features, mem_rep, local_neg_features):
        loss = 0
        # dummy_loss = []
        for i in range(prior_features.size(0)):
            temp = 0.07
            cos = torch.nn.CosineSimilarity()
            criterion = torch.nn.CrossEntropyLoss()
            
            prior_to_image_similarity = cos(mem_rep[None, i, :], prior_features[None, i, :]) / temp
            local_negatives = local_neg_features.detach()
            prior_to_local_negative_similarity = cos(prior_features[None, i, :], local_negatives)/ temp
            similarities = torch.cat((prior_to_image_similarity, prior_to_local_negative_similarity))
            # dummy_loss.append((criterion(similarities[None, :], torch.tensor([0]).to(self.device))).item())
            loss += criterion(similarities[None, :], torch.tensor([0]).to(self.device))
        return loss/prior_features.shape[0]
            
class LocalTripletLoss():
    def __init__(self, device):
        self.device = device 
        self.triplet_loss = torch.nn.TripletMarginLoss(swap=True)
        
    def __call__(self, prior_features, mem_repr, local_neg_features):
        anchor = prior_features
        positive = mem_repr
        negative = local_neg_features
        triplet_loss = self.triplet_loss(anchor, positive, negative)
        
        return triplet_loss/prior_features.shape[0]

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class Logger:
    def __init__(self, file_name, dirc):
        self.file_name = file_name
        index = ['Epoch']
        with open('{}.csv'.format(self.file_name), 'w') as file:
            file.write('Epoch,Loss,Time\n')
        self.writer = SummaryWriter(dirc)  #'./tbx/ss/DiceLoss/'
    
    def embedding(self, feat, label_imgs, meta, epoch):
        if label_imgs is not None:
            label_imgs =label_imgs.cpu().data
            tensor = np.zeros((label_imgs.shape[0], label_imgs.shape[2], label_imgs.shape[3], label_imgs.shape[1]))
            for i in range(label_imgs.shape[0]):
                tensor[i] = np.transpose(label_imgs[i], (1,2,0)) * np.array([[0.3, 0.24, 0.17]]) + np.array([0.56, 0.35, 0.20])
            tensor = np.transpose(tensor, (0,3,2,1))
            # log embeddings Once training is finished in the fc_block features :
            feat = feat.cpu().data
            # get the class labels for each image
            self.writer.add_embedding(feat,
                                   metadata= meta,
                                  label_img=tensor,
                                  global_step=epoch,
                                  tag=''
                                  )
        else:
            feat = feat.cpu().data
            # get the class labels for each image
            self.writer.add_embedding(feat,
                                   metadata= meta,
                                  global_step=epoch,
                                  tag='Contrastive Feature Embedding'
                                  )
            
        return 

    
    def ss_tblog(self, tag, val, epoch):                     
        self.writer.add_scalar(tag=tag, scalar_value=val, global_step=epoch)    
    
    def ss_img(self, tag, val, epoch):
        self.writer.add_images(tag=tag, img_tensor=val, global_step=epoch,dataformats='CHW' )              
        
    def update(self, epoch, loss, lr, train_val='train', acc=None, name=''):
        now = datetime.datetime.now()
        with open('{}.csv'.format(self.file_name), 'a') as file:
            writer = csv.writer(file)
            writer.writerow('{},{:.4f},{}\n'.format(epoch,loss,now))
            # file.write('{},{:.4f},{}\n'.format(epoch,loss,now))
        # self.writer.add_scalar(name+'Lr', lr, epoch)
        if name== '_Lin_':
            self.writer.add_scalar(train_val+name+'Acc', acc, epoch)
        else:
            self.writer.add_scalar(train_val+name+'Loss', loss, epoch)
            
            

def get_lr(optim):
    lr = [group['lr'] for group in optim.param_groups]
    return  lr[0]

def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def print_roc_curve(y_test, y_score, name, n_classes = 2, figsize = (8, 6)):
    # plt.rc('font', family='serif', serif='Times')
    # plt.rc('text', usetex=True)
    # plt.rc('xtick', labelsize=8)
    # plt.rc('ytick', labelsize=8)
    # plt.rc('axes', labelsize=8)
    # width = 3.487
    # height = width / 1.618
    # mpl.use('pdf')
    lw = 2
    y_score = one_hot(y_score,n_classes)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_score[:, i],y_test[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_score.ravel(), y_test.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    fig = plt.figure(figsize=figsize)
    """
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                    ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)
    """
    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    path = os.path.join('visuals/', name +'.eps')
    fig.savefig(path)
    plt.show()
    # return fig
    return


def reset_weights(m):
    '''Reset layer weights, beween folds to prevent leakage '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()
            
 #################################### Alignment and Uniformity #################           
