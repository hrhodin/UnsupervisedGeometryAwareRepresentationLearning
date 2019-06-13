import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import IPython

#import sys
#sys.path.insert(0,'../')

from utils import training as utils_train

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck_noResidual(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.relu(out)

        return out
    
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=16, num_channels = 3, input_width=256, nois_stddev=0, output_key='3D'):
        self.inplanes = 64
        self.output_key = output_key
        self.nois_stddev = nois_stddev
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block,  64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=1)
        
        self.toMap = nn.Sequential(
                        nn.Conv2d(256* block.expansion, 512, kernel_size=1, stride=1, padding=0, bias=True),
                        nn.BatchNorm2d(512),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(512, 128, kernel_size=5, stride=2, padding=0, bias=False),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True)
            )
        
        # size computation of fc input: /16 in resnet, /2 in toMap, -3 in map since no padding but 3x3 (-1) and 5x5 (-2) kernels
        fc_in_width = int(input_width/32)-2 
        fc_in_dimension = 128*fc_in_width*fc_in_width
        self.fc = nn.Linear(fc_in_dimension, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x) # size /2
        x = self.bn1(x)
        x = self.relu(x) 
        x = self.maxpool(x) # size /2

        if self.nois_stddev > 0: x = x + torch.autograd.Variable(torch.randn(x.size()).cuda() * self.nois_stddev)

        x = self.layer1(x)
        
        if self.nois_stddev > 0: x = x + torch.autograd.Variable(torch.randn(x.size()).cuda() * self.nois_stddev)
        
        x = self.layer2(x)# size /2

        if self.nois_stddev > 0: x = x + torch.autograd.Variable(torch.randn(x.size()).cuda() * self.nois_stddev)
        
        x = self.layer3(x)# size /2

        if self.nois_stddev > 0: x = x + torch.autograd.Variable(torch.randn(x.size()).cuda() * self.nois_stddev)

        x = self.layer4(x)

        if self.nois_stddev > 0: x = x + torch.autograd.Variable(torch.randn(x.size()).cuda() * self.nois_stddev)

#        x = self.avgpool(x)
#        x = x.view(x.size(0), -1)
#        x = self.fc(x)

        x = self.toMap(x)
        x = x.view(x.size(0), -1) # 1D per batch
        x = self.fc(x)

        return {self.output_key: x}
    
class ResNet_intermediateOutput(ResNet):

    def __init__(self, block, layers, num_classes=16, input_width=256):
        self.inplanes = 64
        super(ResNet_intermediateOutput, self).__init__(block, layers, num_classes, input_width)

    def forward(self, x):

        x = self.conv1(x) # size /2
        x = self.bn1(x)
        x = self.relu(x) 
        x = self.maxpool(x) # size /2
        
        out0 = x

        x = self.layer1(x) 
        out1 = x
        x = self.layer2(x)# size /2
        out2 = x
        x = self.layer3(x)# size /2
        out3 = x
        x = self.layer4(x)
        out4 = x

#        x = self.avgpool(x)
#        x = x.view(x.size(0), -1)
#        x = self.fc(x)

        x = self.toMap(x)
        x = x.view(x.size(0), -1) # 1D per batch
        x = self.fc(x)

        return [x, out4, out3, out2, out1, out0]


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        utils_train.transfer_partial_weights(model_zoo.load_url(model_urls['resnet18']), model)
        #model.load_state_dict( model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs) # Bottleneck, [3, 4, 6, 3]
    if pretrained:
        utils_train.transfer_partial_weights(model_zoo.load_url(model_urls['resnet50']), model)
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model



def resnet50_intermediate(pretrained=False, **kwargs):
    return ResNet_intermediateOutput(Bottleneck, [3, 4, 6, 3], **kwargs) # Bottleneck, [3, 4, 6, 3]

def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        utils_train.transfer_partial_weights(model_zoo.load_url(model_urls['resnet101']), model)
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        utils_train.transfer_partial_weights(model_zoo.load_url(model_urls['resnet152']), model)
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
