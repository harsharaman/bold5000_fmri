import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.utils import data

class fMRICNN(nn.Module):
    def __init__(self):
        super(fMRICNN, self).__init__()        
        self.conv1 = nn.Conv3d(5, 16, kernel_size=7, stride=2)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=2)
        self.classifier = nn.Conv3d(64, 10, kernel_size=1, stride=2)
        self.tanh = nn.Tanh()
                                                                            
    def forward(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.tanh(x)
        x = self.conv2(x)
        x = self.tanh(x)
        x = self.conv3(x)
        x = self.tanh(x)
        x = self.classifier(x)
        return x
                                                                                                                                                        
net = fMRICNN()
x = torch.randn(35,5,91,109,91)
out = net(x)
print(out.shape)
