import torch
import numpy as np
from models import resnet_low_level

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

class SobelCriterium(torch.nn.Module):
    """
    Approximates horizontal and vertical gradients with the Sobel operator and puts a criterion on these gradient estimates.
    """
    def __init__(self, criterion, weight=1):
        super(SobelCriterium, self).__init__()
        self.weight = weight
        self.criterion = criterion

        kernel_x = np.array([[1, 0, -1], [2,0,-2],  [1, 0,-1]])
        kernel_y = np.array([[1, 2,  1], [0,0, 0], [-1,-2,-1]])

        channels = 3
        kernel_size = 3
        self.conv_x = torch.nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.conv_x.weight = torch.nn.Parameter(torch.from_numpy(kernel_x).float().unsqueeze(0).unsqueeze(0).expand([channels,channels,kernel_size,kernel_size]))
        self.conv_x.weight.requires_grad = False
        self.conv_x.cuda()
        self.conv_y = torch.nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_y.weight = torch.nn.Parameter(torch.from_numpy(kernel_y).float().unsqueeze(0).unsqueeze(0).expand([channels,channels,kernel_size,kernel_size]))
        self.conv_y.weight.requires_grad = False
        self.conv_y.cuda()
        
    def forward(self, pred, label):
        pred_x = self.conv_x.forward(pred)
        pred_y = self.conv_y(pred)
        label_x = self.conv_x(label)
        label_y = self.conv_y(label)

        return self.weight * (self.criterion(pred_x, label_x) + self.criterion(pred_y, label_y))

class ImageNetCriterium(torch.nn.Module):
    """
    Computes difference in the feature space of a NN pretrained on ImageNet
    """
    def __init__(self, criterion, weight=1, do_maxpooling=True):
        super(ImageNetCriterium, self).__init__()
        self.weight = weight
        self.criterion = criterion

        self.net = resnet_low_level.resnet18(pretrained=True, num_channels = 3, do_maxpooling=do_maxpooling)
        for p in self.net.parameters():
            p.requires_grad = False
        self.net.to(device)
        
    def forward(self, pred, label):
        preds_x  = self.net(pred)
        labels_x = self.net(label)
        
        losses = [self.criterion(p, l) for p,l in zip(preds_x,labels_x)]

        return self.weight * sum(losses) / len(losses)
