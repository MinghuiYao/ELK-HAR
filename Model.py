import torch
import torch.nn as nn
import torch.nn.functional as F
from ELK_block import  ELK

'''Base_CNN'''
class Base_CNN(nn.Module):
    def __init__(self, train_shape, category):
        super(Base_CNN, self).__init__()
       
        self.layer = nn.Sequential(
            nn.Conv2d(1,64, (6, 1), (2, 1), (1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(),


            nn.Conv2d(64,128, (6, 1), (2, 1), (1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128,256, (6, 1), (2, 1), (1, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.ada_pool = nn.AdaptiveAvgPool2d((1, train_shape[-1]))
        self.fc = nn.Linear(256*train_shape[-1], category)
        
    def forward(self, x):
        x = self.layer(x)
        x = self.ada_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
'''LK_CNN'''
class LK_CNN(nn.Module):
    def __init__(self, train_shape, category):
        super(LK_CNN, self).__init__()
       
        self.layer = nn.Sequential(
            nn.Conv2d(1,64, (6, 1), (2, 1), (1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(),


            nn.Conv2d(64, 128, (31, 1), (10, 1), (5, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128,256, (6, 1), (2, 1), (1, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.ada_pool = nn.AdaptiveAvgPool2d((1, train_shape[-1]))
        self.fc = nn.Linear(256*train_shape[-1], category)
        
    def forward(self, x):
        x = self.layer(x)
        x = self.ada_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
'''ELK_CNN'''
class ELK_CNN(nn.Module):
    def __init__(self, train_shape, category):
        super(ELK_CNN, self).__init__()
       
        self.layer = nn.Sequential(
            nn.Conv2d(1,64, (6, 1), (2, 1), (1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(),


            ELK(64,128,(31,1),(3,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128,256, (6, 1), (2, 1), (1, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.ada_pool = nn.AdaptiveAvgPool2d((1, train_shape[-1]))
        self.fc = nn.Linear(256*train_shape[-1], category)
        
    def forward(self, x):
        x = self.layer(x)
        x = self.ada_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


