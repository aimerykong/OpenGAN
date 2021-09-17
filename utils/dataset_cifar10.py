import os, random, time, copy
from skimage import io, transform
import numpy as np
import os.path as path
import scipy.io as sio
from scipy import misc
import matplotlib.pyplot as plt
import PIL.Image
import pickle
import skimage.transform 

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler 
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets, models, transforms






class CIFAR_OneClass4Train(Dataset):
    def __init__(self, size=(32,32), set_name='train', 
                 numKnown=6, numTotal=10, runIdx=0, 
                 classLabelIndex=0,
                 path_to_data='/scratch/shuk/dataset/cifar10/cifar-10-batches-py', isOpenset=True,
                 isAugment=True):
        self.classLabelIndex = classLabelIndex
        self.isAugment = isAugment
        self.set_name = set_name
        self.size = size       
        self.numTotal = numTotal
        self.numKnown = numKnown
        self.runIdx = runIdx        
        self.isOpenset = isOpenset
        self.path_to_data = path_to_data
                
        ######### get the data
        # train set            
        curpath = path.join(self.path_to_data, 'data_batch_1')
        with open(curpath, 'rb') as fo:
            curpath = pickle.load(fo, encoding='bytes')

        self.imgList = curpath[b'data'].copy()
        self.labelList = curpath[b'labels'].copy()

        for i in range(2, 6):
            curpath = path.join(path_to_data, 'data_batch_{}'.format(i))
            with open(curpath, 'rb') as fo:
                curpath = pickle.load(fo, encoding='bytes')
            self.imgList = np.concatenate((self.imgList, curpath[b'data'].copy()))
            self.labelList += curpath[b'labels'].copy()
        del curpath
                
        ####### set pre-processing operations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        #self.transform = transforms.Compose([
        #    transforms.RandomCrop(32, padding=4),
        #    transforms.RandomHorizontalFlip(),
        #    transforms.ToTensor(),
        #    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        #])
                
        self.imgList = np.reshape(self.imgList, (self.imgList.shape[0], 3, 32, 32))            
        self.size = size
        self.labelList = np.asarray(self.labelList).astype(np.float32).reshape((-1, 1))
        self.current_set_len = len(self.labelList)
        
        
        ########### shuffle for openset train-test data
        random.seed(0)
        
        self.randShuffleIndexSets = []
        self.OpenSetSplit = [
            [3, 6, 7, 8],
            [1, 2, 4, 6],
            [2, 3, 4, 9],
            [0, 1, 2, 6],
            [4, 5, 6, 9],
            [0, 2, 4, 6, 8, 9], # tinyImageNet
        ]
        for i in range(6):
            tmp = list(range(10))
            tmpCloseset = list(set(tmp)-set(self.OpenSetSplit[i]))
            self.randShuffleIndexSets += [tmpCloseset+self.OpenSetSplit[i]]   
        
        
        self.curShuffleSet = self.randShuffleIndexSets[runIdx]
        self.closesetActualLabels = self.curShuffleSet[:self.numKnown]
        self.opensetActualLabels = self.curShuffleSet[self.numKnown:]
        self.labelmapping = {}
        self.labelmapping_open = {}
        
        for i in range(len(self.closesetActualLabels)):
            self.labelmapping[self.closesetActualLabels[i]] = i
        for j in range(len(self.opensetActualLabels)):
            self.labelmapping_open[self.opensetActualLabels[j]] = self.numKnown + j
            
        self.validList = []
        self.newLabel = []
        for i in range(len(self.labelList)):
            if self.isOpenset:
                if self.labelList[i][0] in self.opensetActualLabels:
                    self.validList += [i]
                    self.newLabel += [self.labelmapping_open[self.labelList[i][0]]]
            else:                
                if self.labelList[i][0] in self.closesetActualLabels:                    
                    tmp_new_label = self.labelmapping[self.labelList[i][0]]                
                    if tmp_new_label==self.classLabelIndex:
                        self.validList += [i]
                        self.newLabel += [tmp_new_label]
                    
        self.imgList = self.imgList[self.validList, :]
        self.labelList = np.asarray(self.newLabel).reshape((len(self.newLabel),1))        
        self.current_set_len = len(self.labelList) 
                              
    def __len__(self):        
        return self.current_set_len
    
    def __getitem__(self, idx):        
        curImage = self.imgList[idx,:]
        curLabel = self.labelList[idx].astype(np.float32)
        
        curImage = PIL.Image.fromarray(curImage.transpose(1,2,0))
        curImage = self.transform(curImage)
        curLabel = torch.from_numpy(curLabel).unsqueeze(0).unsqueeze(0)
        
        return curImage, curLabel  
    





class CIFAR_OPENSET_CLS(Dataset):
    def __init__(self, size=(32,32), set_name='train', 
                 numKnown=6, numTotal=10, runIdx=0, 
                 path_to_data='/scratch/shuk/dataset/cifar10/cifar-10-batches-py', isOpenset=True,
                 isAugment=True):
        
        if set_name=='val':
            set_name = 'test'            
        
        self.isAugment = isAugment
        self.set_name = set_name
        self.size = size       
        self.numTotal = numTotal
        self.numKnown = numKnown
        self.runIdx = runIdx        
        self.isOpenset = isOpenset
        self.path_to_data = path_to_data
                
        ######### get the data
        if self.set_name=='test':
            self.imgList = path.join(self.path_to_data, 'test_batch')
            with open(self.imgList, 'rb') as fo:
                self.imgList = pickle.load(fo, encoding='bytes')
            self.labelList = self.imgList[b'labels'].copy()
            self.imgList = self.imgList[b'data']
        else: # train set            
            curpath = path.join(self.path_to_data, 'data_batch_1')
            with open(curpath, 'rb') as fo:
                curpath = pickle.load(fo, encoding='bytes')

            self.imgList = curpath[b'data'].copy()
            self.labelList = curpath[b'labels'].copy()

            for i in range(2, 6):
                curpath = path.join(path_to_data, 'data_batch_{}'.format(i))
                with open(curpath, 'rb') as fo:
                    curpath = pickle.load(fo, encoding='bytes')
                self.imgList = np.concatenate((self.imgList, curpath[b'data'].copy()))
                self.labelList += curpath[b'labels'].copy()
            del curpath
                
                
        ####### set pre-processing operations
        if self.set_name=='test' or not self.isAugment:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])            
        else:
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
                
        self.imgList = np.reshape(self.imgList, (self.imgList.shape[0], 3, 32, 32))            
        self.size = size
        self.labelList = np.asarray(self.labelList).astype(np.float32).reshape((-1, 1))
        self.current_set_len = len(self.labelList)
        
        
        ########### shuffle for openset train-test data
        random.seed(0)
        
        self.randShuffleIndexSets = []
        self.OpenSetSplit = [
            [3, 6, 7, 8],
            [1, 2, 4, 6],
            [2, 3, 4, 9],
            [0, 1, 2, 6],
            [4, 5, 6, 9],
            [0, 2, 4, 7, 8, 9], # tinyImageNet
        ]
        for i in range(6):
            tmp = list(range(10))
            tmpCloseset = list(set(tmp)-set(self.OpenSetSplit[i]))
            self.randShuffleIndexSets += [tmpCloseset+self.OpenSetSplit[i]]   
        
        #for i in range(10):
        #    a = list(range(10))
        #    random.shuffle(a)
        #    self.randShuffleIndexSets += [a]   
            
            
        self.curShuffleSet = self.randShuffleIndexSets[runIdx]
        self.closesetActualLabels = self.curShuffleSet[:self.numKnown]
        self.opensetActualLabels = self.curShuffleSet[self.numKnown:]
        self.labelmapping = {}
        self.labelmapping_open = {}
        
        for i in range(len(self.closesetActualLabels)):
            self.labelmapping[self.closesetActualLabels[i]] = i
        for j in range(len(self.opensetActualLabels)):
            self.labelmapping_open[self.opensetActualLabels[j]] = self.numKnown + j
            
        
        #self.imgList = np.loadtxt(self.path_to_csv, delimiter=",")
        #self.labelList = np.asfarray(self.imgList[:, :1])
        #self.imgList = np.asfarray(self.imgList[:, 1:]) * self.fac + 0.01       
        
        self.validList = []
        self.newLabel = []
        for i in range(len(self.labelList)):
            if self.isOpenset:
                if self.labelList[i][0] in self.opensetActualLabels:
                    self.validList += [i]
                    self.newLabel += [self.labelmapping_open[self.labelList[i][0]]]
            else:
                if self.labelList[i][0] in self.closesetActualLabels:
                    self.validList += [i]
                    self.newLabel += [self.labelmapping[self.labelList[i][0]]]
                    
        self.imgList = self.imgList[self.validList, :]
        self.labelList = np.asarray(self.newLabel).reshape((len(self.newLabel),1))        
        self.current_set_len = len(self.labelList) 
                              
    def __len__(self):        
        return self.current_set_len
    
    def __getitem__(self, idx):        
        curImage = self.imgList[idx,:]
        curLabel = self.labelList[idx].astype(np.float32)
        
        #if self.isAugment:
        #    curImage = PIL.Image.fromarray(curImage.transpose(1,2,0))
        #    curImage = self.transform(curImage)
        #else:            
        #    curImage = torch.from_numpy(curImage.astype(np.float32))
        
        curImage = PIL.Image.fromarray(curImage.transpose(1,2,0))
        curImage = self.transform(curImage)
        curLabel = torch.from_numpy(curLabel).unsqueeze(0).unsqueeze(0)
        
        '''
        curImage = curImage.astype(np.float32)
        curLabel = curLabel.astype(np.float32)
        
        curImage = torch.from_numpy(curImage)
        curLabel = torch.from_numpy(curLabel).unsqueeze(0).unsqueeze(0)
        '''
        return curImage, curLabel  
    










    
    

class CIFAR10_CLS_full_aug(Dataset):
    def __init__(self, size=(32,32), set_name='train', path_to_data='/scratch/shuk/dataset/cifar10/cifar-10-batches-py', isAugment=True):
        if set_name=='val':
            set_name = 'test'
        self.set_name = set_name
        self.path_to_data = path_to_data
        self.isAugment = isAugment
        
        if self.set_name=='test':
            self.imgList = path.join(self.path_to_data, 'test_batch')
            with open(self.imgList, 'rb') as fo:
                self.imgList = pickle.load(fo, encoding='bytes')
            self.labelList = self.imgList[b'labels'].copy()
            self.imgList = self.imgList[b'data']
        else: # train set            
            curpath = path.join(self.path_to_data, 'data_batch_1')
            with open(curpath, 'rb') as fo:
                curpath = pickle.load(fo, encoding='bytes')

            self.imgList = curpath[b'data'].copy()
            self.labelList = curpath[b'labels'].copy()

            for i in range(2, 6):
                curpath = path.join(path_to_data, 'data_batch_{}'.format(i))
                with open(curpath, 'rb') as fo:
                    curpath = pickle.load(fo, encoding='bytes')
                self.imgList = np.concatenate((self.imgList, curpath[b'data'].copy()))
                self.labelList += curpath[b'labels'].copy()
            del curpath
                
        if self.set_name=='test':
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])            
        else:
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
                
        self.imgList = np.reshape(self.imgList, (self.imgList.shape[0], 3, 32, 32))            
        self.size = size
        self.labelList = np.asarray(self.labelList).astype(np.float32)
        self.current_set_len = len(self.labelList)
        
        
    def __len__(self):        
        return self.current_set_len
    
    def __getitem__(self, idx):        
        curImage = self.imgList[idx]
        curLabel = np.asarray(self.labelList[idx])
        
        if self.isAugment:
            curImage = PIL.Image.fromarray(curImage.transpose(1,2,0))
            curImage = self.transform(curImage)
        else:            
            curImage = torch.from_numpy(curImage)
        
        curLabel = torch.from_numpy(curLabel).unsqueeze(0).unsqueeze(0)

        return curImage, curLabel      
    
    

class CIFAR10_CLS_full(Dataset):
    def __init__(self, size=(32,32), set_name='train', path_to_data='/scratch/shuk/dataset/cifar10/cifar-10-batches-py'):
        if set_name=='val':
            set_name = 'test'
        self.set_name = set_name
        self.path_to_data = path_to_data
        
        if self.set_name=='test':
            self.imgList = path.join(self.path_to_data, 'test_batch')
            with open(self.imgList, 'rb') as fo:
                self.imgList = pickle.load(fo, encoding='bytes')
            self.labelList = self.imgList[b'labels'].copy()
            self.imgList = self.imgList[b'data']
        else: # train set            
            curpath = path.join(self.path_to_data, 'data_batch_1')
            with open(curpath, 'rb') as fo:
                curpath = pickle.load(fo, encoding='bytes')

            self.imgList = curpath[b'data'].copy()
            self.labelList = curpath[b'labels'].copy()

            for i in range(2, 6):
                curpath = path.join(path_to_data, 'data_batch_{}'.format(i))
                with open(curpath, 'rb') as fo:
                    curpath = pickle.load(fo, encoding='bytes')
                self.imgList = np.concatenate((self.imgList, curpath[b'data'].copy()))
                self.labelList += curpath[b'labels'].copy()
            del curpath
        
        self.imgList = np.reshape(self.imgList, (self.imgList.shape[0], 3, 32, 32))            
        self.size = size
        self.fac = 0.99 / 255        
        self.labelList = np.asarray(self.labelList).astype(np.float32)
        self.imgList = self.imgList.astype(np.float32) * self.fac + 0.01
        self.current_set_len = len(self.labelList)
        
    def __len__(self):        
        return self.current_set_len
    
    def __getitem__(self, idx):        
        curImage = self.imgList[idx].astype(np.float32)
        curLabel = np.asarray(self.labelList[idx]).astype(np.float32)

        curImage = torch.from_numpy(curImage)
        curLabel = torch.from_numpy(curLabel).unsqueeze(0).unsqueeze(0)

        return curImage, curLabel  
    
    
    