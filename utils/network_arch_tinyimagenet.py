from __future__ import absolute_import, division, print_function
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
from collections import OrderedDict
from utils.layers import *
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import numpy as np
import os, math
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn




class Discriminator80x80InstNorm(nn.Module):
    def __init__(self, device='cpu', pretrained=False, patchSize=[64, 64], frameStackNumber=3):
        super(Discriminator80x80InstNorm, self).__init__()
        self.device = device
        self.frameStackNumber = frameStackNumber
        self.patchSize = patchSize
        self.outputSize = [patchSize[0]/16, patchSize[1]/16]

        self.discriminator = nn.Sequential(
            # 128-->60
            nn.Conv2d(self.frameStackNumber, 64, kernel_size=5, padding=0, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),

            # 60-->33
            nn.Conv2d(64, 128, kernel_size=5, padding=0, stride=2, bias=False),
            nn.InstanceNorm2d(128, momentum=0.001, affine=False, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 33->
            nn.Conv2d(128, 256, kernel_size=3, padding=0, stride=2, bias=False),
            nn.InstanceNorm2d(256, momentum=0.001, affine=False, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            #
            nn.Conv2d(256, 512, kernel_size=3, padding=0, stride=2, bias=False),
            nn.InstanceNorm2d(512, momentum=0.001, affine=False, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            # final classification for 'real(1) vs. fake(0)'
            nn.Conv2d(512, 1, kernel_size=2, padding=0, stride=2, bias=True),
            nn.Sigmoid()
        )
        
    def forward(self, X):
        return self.discriminator(X)
        
        
        
class Discriminator80x80(nn.Module):
    def __init__(self, device='cpu', pretrained=False, patchSize=[64, 64], frameStackNumber=3):
        super(Discriminator80x80, self).__init__()
        self.device = device
        self.frameStackNumber = frameStackNumber
        self.patchSize = patchSize
        self.outputSize = [patchSize[0]/16, patchSize[1]/16]

        self.discriminator = nn.Sequential(
            # 128-->60
            nn.Conv2d(self.frameStackNumber, 64, kernel_size=5, padding=0, stride=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 60-->33
            nn.Conv2d(64, 128, kernel_size=5, padding=0, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 33->
            nn.Conv2d(128, 256, kernel_size=3, padding=0, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            #
            nn.Conv2d(256, 512, kernel_size=3, padding=0, stride=2, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # final classification for 'real(1) vs. fake(0)'
            nn.Conv2d(512, 1, kernel_size=2, padding=0, stride=2, bias=True),
            nn.Sigmoid()
        )

    def forward(self, X):
        return self.discriminator(X)
    
    
    
class Discriminator70x70(nn.Module):
    def __init__(self, device='cpu', pretrained=False, patchSize=[64, 64], frameStackNumber=3):
        super(Discriminator70x70, self).__init__()
        self.device = device
        self.frameStackNumber = frameStackNumber
        self.patchSize = patchSize
        self.outputSize = [patchSize[0]/16, patchSize[1]/16]

        self.discriminator = nn.Sequential(
            # 128-->60
            nn.Conv2d(self.frameStackNumber, 64, kernel_size=4, padding=0, stride=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 60-->33
            nn.Conv2d(64, 128, kernel_size=4, padding=0, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 33->
            nn.Conv2d(128, 256, kernel_size=4, padding=0, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            #
            nn.Conv2d(256, 512, kernel_size=4, padding=0, stride=2, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # final classification for 'real(1) vs. fake(0)'
            nn.Conv2d(512, 1, kernel_size=2, padding=0, stride=2, bias=True),
            nn.Sigmoid()
        )

    def forward(self, X):
        return self.discriminator(X)


class Discriminator(nn.Module):
    def __init__(self, device='cpu', pretrained=False, patchSize=[64, 64], frameStackNumber=3):
        super(Discriminator, self).__init__()
        self.device = device
        self.frameStackNumber = frameStackNumber
        self.patchSize = patchSize
        self.outputSize = [patchSize[0]/16, patchSize[1]/16]

        self.discriminator = nn.Sequential(
            # 128-->60
            nn.Conv2d(self.frameStackNumber, 64, kernel_size=9, padding=0, stride=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 60-->33
            nn.Conv2d(64, 128, kernel_size=5, padding=0, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 33->
            nn.Conv2d(128, 256, kernel_size=3, padding=0, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=0, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # dropout 
            nn.Dropout(0.7),
            # final classification for 'real(1) vs. fake(0)'
            nn.Conv2d(256, 1, kernel_size=2, padding=0, stride=2, bias=True),
            nn.Sigmoid()
        )

    def forward(self, X):
        return self.discriminator(X)
    
    
    


class GAN_Encoder(nn.Module):
    def __init__(self, embDimension=512):
        super(self.__class__, self).__init__()
    
        self.conv1 = nn.Conv2d(3,       64,     3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64,     128,     1, 2, 0, bias=False)
        self.conv3 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv4 = nn.Conv2d(128,    256,     1, 2, 0, bias=False)
        self.conv5 = nn.Conv2d(256,    256,     3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(256,    512,     1, 2, 0, bias=False)        
        self.conv7 = nn.Conv2d(512,    512,     3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(512,    512,     1, 2, 0, bias=False)
        self.conv9 = nn.Conv2d(512,    512,     3, 1, 1, bias=False)
        self.conv10 = nn.Conv2d(512,   512,      1, 2, 0, bias=False)
        self.conv11 = nn.Conv2d(512,   embDimension,   3, 1, 1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(512)
        self.bn7 = nn.BatchNorm2d(512)
        self.bn8 = nn.BatchNorm2d(512)
        self.bn9 = nn.BatchNorm2d(512)
        self.bn10 = nn.BatchNorm2d(512)
        self.bn11 = nn.BatchNorm2d(embDimension)

        self.apply(weights_init)
        

    def forward(self, x, output_scale=1):
        batch_size = len(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.conv7(x)
        x = self.bn7(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.conv10(x)
        x = self.bn10(x)
        x = nn.LeakyReLU(0.2)(x)

        return x
    
    
class GAN_Decoder(nn.Module):
    def __init__(self, nz=64, ngf=64, nc=3):
        super(GAN_Decoder, self).__init__()
       
        # torch.nn.ConvTranspose2d(
        #    in_channels, out_channels, kernel_size,
        #    stride=1, padding=0, output_padding=0, groups=1, 
        #    bias=True, dilation=1, padding_mode='zeros')
                
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz,     ngf*4,   4,  2,  1,  bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*8) x 2 x 2
            nn.ConvTranspose2d(ngf*4,  ngf*4,   4,  2,  1,  bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d(ngf*4,  ngf*2,   4,  2,  1,  bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose2d(ngf*2,  ngf,     4,  2,  1,  bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 16 x 16
            nn.ConvTranspose2d(ngf,    nc,      4,  2,  1,  bias=True)
            #nn.Tanh()
            # state size. (nc) x 32 x 32
        )            

    def forward(self, x):
        return self.main(x)    
















def weights_init(m):
    classname = m.__class__.__name__
    # TODO: what about fully-connected layers?
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.05)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        

        
        
        
        
class MyDecoder(nn.Module):
    def __init__(self, latent_size=512, input_scale=4, insertConv=False):
        super(self.__class__, self).__init__()
        self.latent_size = latent_size
        self.input_scale = input_scale
        self.fc1 = nn.Linear(latent_size, 512*2*2, bias=False)
        self.insertConv = insertConv
        
        self.conv2_in = nn.ConvTranspose2d(latent_size, 512, 1, stride=1, padding=0, bias=False)        
        self.conv2 = nn.ConvTranspose2d(   512,      512, 4, stride=2, padding=1, bias=False)
        self.conv2_mid = nn.Conv2d(152, 512, 3, 1, 1, bias=False)
        
        self.conv3_in = nn.ConvTranspose2d(latent_size, 512, 1, stride=1, padding=0, bias=False)
        self.conv3 = nn.ConvTranspose2d(   512,      256, 4, stride=2, padding=1, bias=False)
        self.conv3_mid = nn.Conv2d(256, 256, 3, 1, 1, bias=False)        
        
        self.conv4_in = nn.ConvTranspose2d(latent_size, 256, 1, stride=1, padding=0, bias=False)        
        self.conv4 = nn.ConvTranspose2d(   256,      128, 4, stride=2, padding=1, bias=False)
        self.conv4_mid = nn.Conv2d(128, 128, 3, 1, 1, bias=False)        
                
        self.conv5 = nn.ConvTranspose2d(   128,        128, 4, stride=2, padding=1, bias=False)
        self.conv5_mid = nn.Conv2d(128, 128, 3, 1, 1, bias=False)        
        
        
        self.conv6 = nn.ConvTranspose2d(   128,        3, 4, stride=2, padding=1, bias=True)
        

        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(512)
        self.bn2_mid = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn3_mid = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn4_mid = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn5_mid = nn.BatchNorm2d(128)

        self.apply(weights_init)
        self.cuda()

        
    def forward(self, x):
        input_scale=self.input_scale
        batch_size = x.shape[0]
        
        if input_scale <= 1:
            x = self.fc1(x)
            x = x.resize(batch_size, 512, 2, 2)

        # 512 x 2 x 2
        if input_scale == 2:
            x = x.view(batch_size, self.latent_size, 2, 2)
            x = self.conv2_in(x)
        if input_scale <= 2:
            x = self.conv2(x)
            x = nn.LeakyReLU()(x)
            x = self.bn2(x)
            if self.insertConv:
                x = self.conv2_mid(x)
                x = nn.LeakyReLU()(x)
                x = self.bn2_mid(x)

        # 512 x 4 x 4
        if input_scale == 4:
            x = x.view(batch_size, self.latent_size, 4, 4)
            x = self.conv3_in(x)
        if input_scale <= 4:
            x = self.conv3(x)
            x = nn.LeakyReLU()(x)
            x = self.bn3(x)
            if self.insertConv:
                x = self.conv3_mid(x)
                x = nn.LeakyReLU()(x)
                x = self.bn3_mid(x)
        
        
        # 256 x 8 x 8
        if input_scale == 8:
            x = x.view(batch_size, self.latent_size, 8, 8)
            x = self.conv4_in(x)
        if input_scale <= 8:
            x = self.conv4(x)
            x = nn.LeakyReLU()(x)
            x = self.bn4(x)
            if self.insertConv:
                x = self.conv4_mid(x)
                x = nn.LeakyReLU()(x)
                x = self.bn4_mid(x)
        
        
        # 128 x 16 x 16
        x = self.conv5(x)
        x = nn.LeakyReLU()(x)
        x = self.bn5_mid(x)
        
        # 3 x 32 x 32
        #x = nn.Sigmoid()(x)
        
        x = self.conv6(x)
        return x
        
        
        


class MySingleBigDecoder(nn.Module):
    def __init__(self, latent_size=512, input_scale=4, insertConv=False, nClasses=200):
        super(self.__class__, self).__init__()
        self.latent_size = latent_size
        self.input_scale = input_scale
        self.fc1 = nn.Linear(latent_size, 512*2*2, bias=False)
        self.insertConv = insertConv
        self.nClasses = nClasses
        
        self.conv2_in = nn.ConvTranspose2d(latent_size, 512, 1, stride=1, padding=0, bias=False)        
        self.conv2 = nn.ConvTranspose2d(   512,      512, 4, stride=2, padding=1, bias=False)
        self.conv2_mid = nn.Conv2d(152, 512, 3, 1, 1, bias=False)
        
        self.conv3_in = nn.ConvTranspose2d(latent_size, 512, 1, stride=1, padding=0, bias=False)
        self.conv3 = nn.ConvTranspose2d(   512,      256, 4, stride=2, padding=1, bias=False)
        self.conv3_mid = nn.Conv2d(256, 256, 3, 1, 1, bias=False)        
        
        self.conv4_in = nn.ConvTranspose2d(latent_size, 256, 1, stride=1, padding=0, bias=False)        
        self.conv4 = nn.ConvTranspose2d(   256,      128, 4, stride=2, padding=1, bias=False)
        self.conv4_mid = nn.Conv2d(128, 128, 3, 1, 1, bias=False)        
                
        self.conv5 = nn.ConvTranspose2d(   128,        128, 4, stride=2, padding=1, bias=False)
        self.conv5_mid = nn.Conv2d(128, 128, 3, 1, 1, bias=False)        
        
        
        self.conv6 = nn.ConvTranspose2d(   128, 3*nClasses, 4, stride=2, padding=1, bias=True)
        

        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(512)
        self.bn2_mid = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn3_mid = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn4_mid = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn5_mid = nn.BatchNorm2d(128)

        self.apply(weights_init)
        self.cuda()

        
    def forward(self, x):
        input_scale=self.input_scale
        batch_size = x.shape[0]
        
        if input_scale <= 1:
            x = self.fc1(x)
            x = x.resize(batch_size, 512, 2, 2)

        # 512 x 2 x 2
        if input_scale == 2:
            x = x.view(batch_size, self.latent_size, 2, 2)
            x = self.conv2_in(x)
        if input_scale <= 2:
            x = self.conv2(x)
            x = nn.LeakyReLU()(x)
            x = self.bn2(x)
            if self.insertConv:
                x = self.conv2_mid(x)
                x = nn.LeakyReLU()(x)
                x = self.bn2_mid(x)

        # 512 x 4 x 4
        if input_scale == 4:
            x = x.view(batch_size, self.latent_size, 4, 4)
            x = self.conv3_in(x)
        if input_scale <= 4:
            x = self.conv3(x)
            x = nn.LeakyReLU()(x)
            x = self.bn3(x)
            if self.insertConv:
                x = self.conv3_mid(x)
                x = nn.LeakyReLU()(x)
                x = self.bn3_mid(x)
        
        
        # 256 x 8 x 8
        if input_scale == 8:
            x = x.view(batch_size, self.latent_size, 8, 8)
            x = self.conv4_in(x)
        if input_scale <= 8:
            x = self.conv4(x)
            x = nn.LeakyReLU()(x)
            x = self.bn4(x)
            if self.insertConv:
                x = self.conv4_mid(x)
                x = nn.LeakyReLU()(x)
                x = self.bn4_mid(x)
        
        
        # 128 x 16 x 16
        x = self.conv5(x)
        x = nn.LeakyReLU()(x)
        x = self.bn5_mid(x)
        
        # 3 x 32 x 32
        #x = nn.Sigmoid()(x)
        
        x = self.conv6(x)
        return x
        
                
        
        
        
        
class MyDecoder_noBN(nn.Module):
    def __init__(self, latent_size=512, input_scale=4, insertConv=False):
        super(self.__class__, self).__init__()
        self.latent_size = latent_size
        self.input_scale = input_scale
        self.fc1 = nn.Linear(latent_size, 512*2*2, bias=False)
        self.insertConv = insertConv
        
        self.conv2_in = nn.ConvTranspose2d(latent_size, 512, 1, stride=1, padding=0, bias=True)        
        self.conv2 = nn.ConvTranspose2d(   512,      512, 4, stride=2, padding=1, bias=True)
        self.conv2_mid = nn.Conv2d(152, 512, 3, 1, 1, bias=True)
        
        self.conv3_in = nn.ConvTranspose2d(latent_size, 512, 1, stride=1, padding=0, bias=True)
        self.conv3 = nn.ConvTranspose2d(   512,      256, 4, stride=2, padding=1, bias=True)
        self.conv3_mid = nn.Conv2d(256, 256, 3, 1, 1, bias=True)        
        
        self.conv4_in = nn.ConvTranspose2d(latent_size, 256, 1, stride=1, padding=0, bias=True)        
        self.conv4 = nn.ConvTranspose2d(   256,      128, 4, stride=2, padding=1, bias=True)
        self.conv4_mid = nn.Conv2d(128, 128, 3, 1, 1, bias=True)
                
        self.conv5 = nn.ConvTranspose2d(   128,        3, 4, stride=2, padding=1, bias=True)
        self.conv5_mid = nn.Conv2d(128, 128, 3, 1, 1, bias=False)        
        
        self.conv6 = nn.ConvTranspose2d(   128,        3, 4, stride=2, padding=1, bias=True)

        #self.bn1 = nn.BatchNorm2d(512)
        #self.bn2 = nn.BatchNorm2d(512)
        #self.bn2_mid = nn.BatchNorm2d(512)
        #self.bn3 = nn.BatchNorm2d(256)
        #self.bn3_mid = nn.BatchNorm2d(256)
        #self.bn4 = nn.BatchNorm2d(128)
        #self.bn4_mid = nn.BatchNorm2d(128)

        self.apply(weights_init)
        self.cuda()

    def forward(self, x):
        input_scale=self.input_scale
        batch_size = x.shape[0]
        
        if input_scale <= 1:
            x = self.fc1(x)
            x = x.resize(batch_size, 512, 2, 2)

        # 512 x 2 x 2
        if input_scale == 2:
            x = x.view(batch_size, self.latent_size, 2, 2)
            x = self.conv2_in(x)
        if input_scale <= 2:
            x = self.conv2(x)
            x = nn.LeakyReLU()(x)
            #x = self.bn2(x)
            if self.insertConv:
                x = self.conv2_mid(x)
                x = nn.LeakyReLU()(x)
                #x = self.bn2_mid(x)

        # 512 x 4 x 4
        if input_scale == 4:
            x = x.view(batch_size, self.latent_size, 4, 4)
            x = self.conv3_in(x)
            x = nn.LeakyReLU()(x)
        if input_scale <= 4:
            x = self.conv3(x)
            x = nn.LeakyReLU()(x)
            #x = self.bn3(x)
            if self.insertConv:
                x = self.conv3_mid(x)
                x = nn.LeakyReLU()(x)
                #x = self.bn3_mid(x)
        
        
        # 256 x 8 x 8
        if input_scale == 8:
            x = x.view(batch_size, self.latent_size, 8, 8)
            x = self.conv4_in(x)
        if input_scale <= 8:
            x = self.conv4(x)
            x = nn.LeakyReLU()(x)
            #x = self.bn4(x)
            if self.insertConv:
                x = self.conv4_mid(x)
                x = nn.LeakyReLU()(x)
                #x = self.bn4_mid(x)
        
        
        # 128 x 16 x 16
        x = self.conv5(x)
        x = nn.LeakyReLU()(x)
        #x = self.bn5_mid(x)
        
        # 3 x 32 x 32
        #x = nn.Sigmoid()(x)
        
        x = self.conv6(x)
        return x
        
        
        
                
        
        


class classifier32(nn.Module):
    def __init__(self, latent_size=100, num_classes=2, batch_size=64, return_feat=True):
        super(self.__class__, self).__init__()
        self.return_feat = return_feat
        
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3,       64,     3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64,      64,     3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64,     128,     3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)

        self.conv7 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128*4*4, num_classes)
        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)

        self.apply(weights_init)
        self.cuda()

    def forward(self, x, return_features=False):
        batch_size = len(x)

        x = self.dr1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr2(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr3(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = nn.LeakyReLU(0.2)(x)

        x = x.view(batch_size, -1)
        if self.return_feat:
            return x
        x = self.fc1(x)
        return x
    
    

class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers=18, isPretrained=False, isGrayscale=False, embDimension=128, poolSize=4):
        super(ResnetEncoder, self).__init__()
        self.path_to_model = '../models'
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.isGrayscale = isGrayscale
        self.isPretrained = isPretrained
        self.embDimension = embDimension
        self.poolSize = poolSize
        self.featListName = ['layer0', 'layer1', 'layer2', 'layer3', 'layer4']
        
        resnets = {
            18: models.resnet18, 
            34: models.resnet34,
            50: models.resnet50, 
            101: models.resnet101,
            152: models.resnet152}
        
        resnets_pretrained_path = {
            18: 'resnet18-5c106cde.pth', 
            34: 'resnet34.pth',
            50: 'resnet50.pth',
            101: 'resnet101.pth',
            152: 'resnet152.pth'}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(
                num_layers))

        self.encoder = resnets[num_layers]()
        
        if self.embDimension>0:
            self.encoder.linear =  nn.Linear(self.num_ch_enc[-1], self.embDimension)
        
        if self.isPretrained:
            print("using pretrained model")
            self.encoder.load_state_dict(
                torch.load(os.path.join(self.path_to_model, resnets_pretrained_path[num_layers])))
        
        #if self.isGrayscale:
        #    self.encoder.conv1 = nn.Conv2d(
        #        1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        #else:
        #    self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        
        x = self.encoder.conv1(input_image)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        self.features.append(x)
        
        #x = self.encoder.layer1(self.encoder.maxpool(x)) # 
        x = self.encoder.layer1(x) # self.encoder.maxpool(x)
        self.features.append(x)
        #print('layer1: ', x.shape)
        
        x = self.encoder.layer2(x)
        self.features.append(x)
        #print('layer2: ', x.shape)
        
        x = self.encoder.layer3(x) 
        self.features.append(x)
        #print('layer3: ', x.shape)       
        
        x = self.encoder.layer4(x)
        self.features.append(x)
        #print('layer4: ', x.shape)
        
        x = F.avg_pool2d(x, self.poolSize)
        #print('global pool: ', x.shape)
        
        x = x.view(x.size(0), -1)
        #print('reshape: ', x.shape)
        
        if self.embDimension>0:
            x = self.encoder.linear(x)
        #print('final: ', x.shape)
        return x
    
    
    
class TinyImageNet_ClsNet(nn.Module):
    def __init__(self, nClass=10, layerList=(64, 32)):
        super(TinyImageNet_ClsNet, self).__init__()
        
        self.nClass = nClass
        self.layerList = layerList
        self.linearLayers = OrderedDict()
        self.relu = nn.ReLU()
        i=-1
        for i in range(len(layerList)-1):
            self.linearLayers[i] = nn.Linear(self.layerList[i], self.layerList[i+1])            
        self.linearLayers[i+1] = nn.Linear(self.layerList[-1], self.nClass)        
        self.mnist_clsnet = nn.ModuleList(list(self.linearLayers.values()))
        
    def forward(self, x):  
        i = -1
        for i in range(len(self.layerList)-1):
            x = self.linearLayers[i](x)
            x = self.relu(x)
        x = self.linearLayers[i+1](x)            
        return x
    

    
    
    
class TinyImageNet_Decoder(nn.Module):
    def __init__(self, embDimension=128, layerList=(256, 512, 3*1024*1024), imgSize=[3,32,32], 
                 isReshapeBack=True, reluFirst=False):
        super(TinyImageNet_Decoder, self).__init__()
        
        self.imgSize = imgSize
        self.embDimension = embDimension
        self.layerList = layerList
        self.linearLayers = OrderedDict()
        self.relu = nn.ReLU()
        self.isReshapeBack = isReshapeBack
        self.reluFirst = reluFirst
        
        self.linearLayers[0] = nn.Linear(self.embDimension, self.layerList[0])
        for i in range(1, len(layerList)):
            self.linearLayers[i] = nn.Linear(self.layerList[i-1], self.layerList[i])            
              
        self.mnist_decoder = nn.ModuleList(list(self.linearLayers.values()))
        
    def forward(self, x): 
        self.featList = []
        
        if self.reluFirst:
            x = self.relu(x)            
        x = self.linearLayers[0](x)
        self.featList.append(x)
        
        for i in range(1, len(self.layerList)):
            x = self.relu(x)
            x = self.linearLayers[i](x)
            self.featList.append(x)
            
        if self.isReshapeBack:
            x = x.view(x.size(0), self.imgSize[0], self.imgSize[1], self.imgSize[2])
        
        return x

    
    
class CondEncoder(nn.Module):
    def __init__(self, num_classes=200, dimension=128, device='cpu'):
        super(self.__class__, self).__init__()
        self.num_classes = num_classes               
        self.dimension = dimension

        self.fc1 = nn.Linear(num_classes, num_classes)
        self.fc2 = nn.Linear(num_classes, dimension)
        self.fc3 = nn.Linear(dimension, dimension)
        self.device = device
        
    def forward(self, input, indicator):
        batch_size = len(input)
        x = torch.zeros(batch_size, self.num_classes).to(self.device)
        x[:, indicator] = 1
        x = x.to(self.device)
        
        x = self.fc1(x)        
        x = nn.LeakyReLU(0.2)(x)
        x = self.fc2(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.fc3(x)
        return x       