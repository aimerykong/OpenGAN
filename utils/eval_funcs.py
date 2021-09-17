import os, random, time, copy
from skimage import io, transform
import numpy as np
import os.path as path
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image
import PIL.Image
from sklearn.metrics import roc_curve, roc_auc_score, f1_score
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler 
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import models, transforms

import sklearn.metrics 

def F_measure(preds, labels, openset=False, theta=None):
    if openset:
        # f1 score for openset evaluation
        true_pos = 0.
        false_pos = 0.
        false_neg = 0.        
        for i in range(len(labels)):
            true_pos += 1 if preds[i] == labels[i] and labels[i] != -1 else 0
            false_pos += 1 if preds[i] != labels[i] and labels[i] != -1 else 0
            false_neg += 1 if preds[i] != labels[i] and labels[i] == -1 else 0

        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        return 2 * ((precision * recall) / (precision + recall + 1e-12))
    else: # Regular f1 score        
        return f1_score(labels, preds, average='macro')

# 
# ref:  https://github.com/lwneal/counterfactual-open-set/blob/master/generativeopenset/evaluation.py
class ClassCentroids(nn.Module):
    def __init__(self, num_classes=10, feat_dim=2, device='cpu'):
        super(ClassCentroids, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim        
        self.centers = torch.randn(self.num_classes, self.feat_dim)
        #self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        self.device = device
        if self.device!='cpu':
            self.centers.to(self.device)

    def forward(self, x, labels):
        batch_size = x.size(0)
        # ||x-y||_2 = (x-y)^2 = x^2 + y^2 - 2xy
        # This part of the calculation is “x^2+y^2”        
        distmat =  torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        # This part is "x^2+y^2 - 2xy"
        distmat.addmm_(1, -2, x, self.centers.t())
        
        classes = torch.arange(self.num_classes).long().to(self.device)
        if self.device!='cpu':
            classes = classes.to(self.device)
            
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        self.curDistMat = distmat
        
        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

class CosCentroid(nn.Module):
    def __init__(self, num_classes=10, feat_dim=2, device='cpu'):
        super(CosCentroid, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim        
        self.centers = torch.randn(self.num_classes, self.feat_dim)
        self.device = device
        #self.centers = F.normalize(self.centers, p=2, dim=1)
        if self.device!='cpu':
            self.centers.to(self.device)

    def forward(self, x, label=0):
        x = F.normalize(x, p=2, dim=1)
        distmat = torch.zeros((x.shape[0], self.centers.shape[0])).to(self.device)
        distmat.addmm_(0, -1, x, self.centers.t())
        self.curDistMat = distmat        
        return self.curDistMat 
    
    
    
def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    m = np.mean(X, 0)
    X = X - np.tile(m, (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    P = M[:, 0:no_dims]
    Y = np.dot(X, P)
    return Y, m, P





def FetchFromSingleImage(curImg, cropSize=64, scaleList=[64, 78, 96, 128]):
    imgBatchList = []
    
    for curSize in scaleList:
        curImg = curImg.resize((curSize, curSize))        
        curTransform = transforms.Compose([
            transforms.TenCrop(cropSize, vertical_flip=False),
            transforms.Lambda(lambda crops: torch.stack([
                transforms.ToTensor()(crop) for crop in crops])), # returns a 4D tensor
            transforms.Lambda(lambda crops: torch.stack([
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(crop) for crop in crops])),
            ])
        imgBatchList += list(curTransform(curImg).unsqueeze(0))


    curImg = curImg.resize((cropSize,cropSize))
    curTransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    imgBatchList += [curTransform(curImg).unsqueeze(0).clone()]
    imgBatchList += [curTransform(curImg.transpose(PIL.Image.FLIP_LEFT_RIGHT)).unsqueeze(0).clone()]
    imgBatchList = torch.cat(imgBatchList, 0)
    return imgBatchList



class CustomizedPoolList(nn.Module):
    def __init__(self, poolSizeList=[32,32,16,8,4], poolType='max'):
        super(CustomizedPoolList, self).__init__()
        
        self.poolSizeList = poolSizeList
        self.poolType = poolType
        #self.linearLayers = OrderedDict()
        self.relu = nn.ReLU()
        #self.mnist_clsnet = nn.ModuleList(list(self.linearLayers.values()))
        
    def forward(self, feaList):
        x = []
        if self.poolType=='max':
            for i in range(len(self.poolSizeList)):
                if self.poolSizeList[i]>0:
                    x += [F.max_pool2d(feaList[i], self.poolSizeList[i])]
        elif self.poolType=='avg':
            for i in range(len(self.poolSizeList)):
                if self.poolSizeList[i]>0:
                    x += [F.avg_pool2d(feaList[i], self.poolSizeList[i])]
        
        x = torch.cat(x, 1)  
        x = x.view(x.shape[0], -1)
        return x



class weightedL1Loss(nn.Module):
    def __init__(self, weight=1):
        # mean over all
        super(weightedL1Loss, self).__init__()        
        self.loss = nn.L1Loss()
        self.weight = weight
        
    def forward(self, inputs, target): 
        lossValue = self.weight * self.loss(inputs, target)
        return lossValue 




class MetricLoss(nn.Module):
    """inner-class compactness, aka Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, 
                 weightCompactness=0.2, 
                 weightInner=1,
                 weightInter=1.,
                 marginAlpha=0.2,
                 sepMultiplier=3,
                 device='cpu'):
        super(MetricLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.weightCompactness = weightCompactness
        self.weightInner = weightInner
        self.weightInter = weightInter        
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        self.device = device
        if self.device!='cpu':
            self.centers.to(self.device)
        self.curDistMat = 0
        self.lossInner = 0
        self.lossInter = 0
        self.marginAlpha = marginAlpha
        self.sepMultiplier = sepMultiplier
        self.classes = torch.arange(self.num_classes).long().to(self.device)
        
    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        # ||x-y||_2 = (x-y)^2 = x^2 + y^2 - 2xy
        # This part of the calculation is “x^2+y^2”        
        distmat =  torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        # This part is "x^2+y^2 - 2xy"
        distmat.addmm_(1, -2, x, self.centers.t())
                    
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(self.classes.expand(batch_size, self.num_classes))

        self.curDistMat = distmat
        #print('self.curDistMat:  ', self.curDistMat.shape)
        
        # inner loss
        dist = distmat * mask.float()
        self.lossInner = (dist-self.marginAlpha).clamp(min=0)
        self.lossInner = self.lossInner.mean()*self.weightInner  # / batch_size
        
        # compactness loss
        loss = dist.clamp(min=1e-12, max=1e+12).mean() / batch_size
        
        # inter loss
        # distance between centroids should be at least three times larger than the defined margin alpha        
        self.lossInter = torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, self.num_classes) + torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, self.num_classes).t()
        self.lossInter.addmm_(1, -2, self.centers, self.centers.t())        
        tmpMask = 1-torch.eye(self.num_classes).float().to(self.device)  
        #tmpMask = tmpMask.reshape((1, self.num_classes, self.num_classes)) 
        #tmpMask = tmpMask.repeat(batch_size, 1, 1).to(self.device)       
        self.lossInter = (self.marginAlpha*self.sepMultiplier-self.lossInter).clamp(min=0)
        self.lossInter = self.lossInter*tmpMask
        self.lossInter = self.lossInter.sum()*self.weightInter
        
        return loss*self.weightCompactness
    
    
    

def evaluate_openset(scores_closeset, scores_openset):    
    y_true = np.array([0] * len(scores_closeset) + [1] * len(scores_openset))
    y_discriminator = np.concatenate([scores_closeset, scores_openset])
    auc_d, roc_to_plot = plot_roc(y_true, y_discriminator, 'Discriminator ROC')
    return auc_d, roc_to_plot


def plot_roc(y_true, y_score, title="Receiver Operating Characteristic", **options):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)
    roc_to_plot = {'tp':tpr, 'fp':fpr, 'thresh':thresholds, 'auc_score':auc_score}
    #plot = plot_xy(fpr, tpr, x_axis="False Positive Rate", y_axis="True Positive Rate", title=title)
    #if options.get('roc_output'):
    #    print("Saving ROC scores to file")
    #    np.save(options['roc_output'], (fpr, tpr))
    #return auc_score, plot, roc_to_plot
    return auc_score, roc_to_plot


def plot_xy(x, y, x_axis="X", y_axis="Y", title="Plot"):
    df = pd.DataFrame({'x': x, 'y': y})
    plot = df.plot(x='x', y='y')

    plot.grid(b=True, which='major')
    plot.grid(b=True, which='minor')
    
    plot.set_title(title)
    plot.set_ylabel(y_axis)
    plot.set_xlabel(x_axis)
    return plot


def backup_Weibull():
    print("Weibull: computing features for all correctly-classified training data")
    activation_vectors = {}
    for images, labels in dataloader_train_closeset:         
        images = images.to(device)
        labels = labels.type(torch.long).view(-1).to(device)

        embFeature = encoder(images)
        logits = clsModel(embFeature)
        #logits =  F.softmax(logits, dim=1)

        correctly_labeled = (logits.data.max(1)[1] == labels)
        labels_np = labels.cpu().numpy()
        logits_np = logits.data.cpu().numpy()
        for i, label in enumerate(labels_np):
            if not correctly_labeled[i]:
                continue
            if label not in activation_vectors:
                activation_vectors[label] = []
            activation_vectors[label].append(logits_np[i])

    print("Computed activation_vectors for {} known classes".format(len(activation_vectors)))
    for class_idx in activation_vectors:
        print("Class {}: {} images".format(class_idx, len(activation_vectors[class_idx])))    
        
    # Compute a mean activation vector for each class
    print("Weibull computing mean activation vectors...")
    mean_activation_vectors = {}
    for class_idx in activation_vectors:
        mean_activation_vectors[class_idx] = np.array(activation_vectors[class_idx]).mean(axis=0)        
        
    WEIBULL_TAIL_SIZE = 20
    # Initialize one libMR Wiebull object for each class
    print("Fitting Weibull to distance distribution of each class")
    weibulls = {}
    for class_idx in activation_vectors:
        distances = []
        mav = mean_activation_vectors[class_idx]
        for v in activation_vectors[class_idx]:
            distances.append(np.linalg.norm(v - mav))
        mr = libmr.MR()
        tail_size = min(len(distances), WEIBULL_TAIL_SIZE)
        mr.fit_high(distances, tail_size)
        weibulls[class_idx] = mr
        print("Weibull params for class {}: {}".format(class_idx, mr.get_params()))
        
        
    # Apply Weibull score to every logit
    weibull_scores_closeset = []
    logits_closeset = []
    classes = activation_vectors.keys()
    for images, labels in dataloader_test_closeset:
        images = images.to(device)
        labels = labels.type(torch.long).view(-1).to(device)    
        embFeature = encoder(images)
        batch_logits = clsModel(embFeature).data.cpu().numpy()    
        batch_weibull = np.zeros(shape=batch_logits.shape)
        for activation_vector in batch_logits:
            weibull_row = np.ones(len(classes))
            for class_idx in classes:
                mav = mean_activation_vectors[class_idx]
                dist = np.linalg.norm(activation_vector - mav)
                weibull_row[class_idx] = 1 - weibulls[class_idx].w_score(dist)
            weibull_scores_closeset.append(weibull_row)
            logits_closeset.append(activation_vector)

    weibull_scores_closeset = np.array(weibull_scores_closeset)
    logits_closeset = np.array(logits_closeset)
    openmax_scores_closeset = -np.log(np.sum(np.exp(logits_closeset * weibull_scores_closeset), axis=1))        
        

    # Apply Weibull score to every logit
    weibull_scores_openset = []
    logits_openset = []
    classes = activation_vectors.keys()
    for images, labels in dataloader_test_openset:
        images = images.to(device)
        labels = labels.type(torch.long).view(-1).to(device)    
        embFeature = encoder(images)
        batch_logits = clsModel(embFeature).data.cpu().numpy()    
        batch_weibull = np.zeros(shape=batch_logits.shape)
        for activation_vector in batch_logits:
            weibull_row = np.ones(len(classes))
            for class_idx in classes:
                mav = mean_activation_vectors[class_idx]
                dist = np.linalg.norm(activation_vector - mav)
                weibull_row[class_idx] = 1 - weibulls[class_idx].w_score(dist)
            weibull_scores_openset.append(weibull_row)
            logits_openset.append(activation_vector)

    weibull_scores_openset = np.array(weibull_scores_openset)
    logits_openset = np.array(logits_openset)
    openmax_scores_openset = -np.log(np.sum(np.exp(logits_openset * weibull_scores_openset), axis=1))        