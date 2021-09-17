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
import csv
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler 
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets, models, transforms




class TINYIMAGENET(Dataset):
    def __init__(self, size=(64,64), set_name='train',
                 path_to_data='/scratch/shuk/dataset/tiny-imagenet-200', 
                 isAugment=True):
        
        self.path_to_data = path_to_data        
        self.mapping_name2id = {}
        self.mapping_id2name = {}
        with open(path.join(self.path_to_data, 'wnids.txt')) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=' ')
            idx = 0
            for row in csv_reader:
                self.mapping_id2name[idx] = row[0]
                self.mapping_name2id[row[0]] = idx
                idx += 1
        
        
        if set_name=='test':  set_name = 'val'
        
        self.size = size
        self.set_name = set_name
        self.path_to_data = path_to_data
        self.isAugment = isAugment
        
        self.imageNameList  = []
        self.className = []
        self.labelList = []
        self.mappingLabel2Name = dict()
        curLabel = 0

        
        if self.set_name == 'val':
            with open(path.join(self.path_to_data, 'val', 'val_annotations.txt')) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter='\t')
                line_count = 0
                for row in csv_reader:
                    self.imageNameList += [path.join(self.path_to_data, 'val', 'images', row[0])]
                    self.labelList += [self.mapping_name2id[row[1]]]
        else: # 'train'
            self.current_class_dir = path.join(self.path_to_data, self.set_name)
            for curClass in os.listdir(self.current_class_dir):                 
                if curClass[0]=='.':   continue
                
                curLabel = self.mapping_name2id[curClass]
                for curImg in os.listdir(path.join(self.current_class_dir, curClass, 'images')):
                    if curImg[0]=='.':    continue
                    self.labelList += [curLabel]
                    self.imageNameList += [path.join(self.path_to_data, self.set_name, curClass, 'images', curImg)]

        
        self.current_set_len = len(self.labelList)
        
        if self.set_name=='test' or self.set_name=='val' or  not self.isAugment:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
            ])            # ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        else:
            self.transform = transforms.Compose([
                transforms.RandomCrop(self.size[0], padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
            ])
        
    def __len__(self):        
        return self.current_set_len
    
    def __getitem__(self, idx):
        curLabel = np.asarray(self.labelList[idx])
        curImage = self.imageNameList[idx]
        curImage = PIL.Image.open(curImage).convert('RGB')
        curImage = self.transform(curImage)
        
        #print(idx, curLabel)
        
        #curLabel = torch.tensor([curLabel]).unsqueeze(0).unsqueeze(0)

        return curImage, curLabel