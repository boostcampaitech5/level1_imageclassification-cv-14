import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
from torchvision import models
import timm

    
    
class EfficientB4(nn.Module):
    def __init__(self, num_classes = 18):
        super(EfficientB4, self).__init__()
        self.effnet = timm.create_model('efficientnet_b4', pretrained=True)
        in_features = self.effnet.classifier.in_features
        self.effnet.classifier = nn.Linear(in_features , num_classes)

    def forward(self, x):
        x = self.effnet(x)
        return x
    
    
class EfficientB0(nn.Module):
    def __init__(self, num_classes = 18):
        super(EfficientB0, self).__init__()
        self.effnet = timm.create_model('efficientnet_b0', pretrained=True)
        in_features = self.effnet.classifier.in_features
        self.effnet.classifier = nn.Linear(in_features , num_classes)

    def forward(self, x):
        x = self.effnet(x)
        return x

class efficientnetv2_s(nn.Module):
    def __init__(self, num_classes = 18):
        super(efficientnetv2_s, self).__init__()
        self.effnet = timm.create_model('efficientnetv2_s', pretrained=True)
        in_features = self.effnet.classifier.in_features
        self.effnet.classifier = nn.Linear(in_features , num_classes)

    def forward(self, x):
        x = self.effnet(x)
        return x
    
class EfficientB2(nn.Module):
    def __init__(self, num_classes = 18):
        super(EfficientB2, self).__init__()
        self.effnet = timm.create_model('efficientnet_b2', pretrained=True)
        in_features = self.effnet.classifier.in_features
        self.effnet.classifier = nn.Linear(in_features , num_classes)

    def forward(self, x):
        x = self.effnet(x)
        return x
    
class Resnet18(nn.Module):
    def __init__(self, num_classes = 18):
        super(Resnet18, self).__init__()
        self.resnet = models.resnet18(pretrained = True)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x
    

class Resnet101(nn.Module):
    def __init__(self, num_classes = 18):
        super(Resnet101, self).__init__()
        self.resnet = models.resnet101(pretrained = True)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x
    

class Resnet34(nn.Module):
    def __init__(self, num_classes = 18):
        super(Resnet34, self).__init__()
        self.resnet = models.resnet34(pretrained = True)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x
    
class Resnet50(nn.Module):
    def __init__(self, num_classes = 18):
        super(Resnet50, self).__init__()
        self.resnet = models.resnet50(pretrained = True)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x
    
class Resnet152(nn.Module):
    def __init__(self, num_classes = 18):
        super(Resnet152, self).__init__()
        self.resnet = models.resnet152(pretrained = True)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x
    

class Densenet201(nn.Module):
    def __init__(self, num_classes = 18):
        super(Densenet201, self).__init__()
        self.model = models.densenet201(pretrained = True).features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(1920, out_features=num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.avgpool(x).view(x.size()[0], -1)
        x = self.fc(x)
        return x
    

class Densenet121(nn.Module):
    def __init__(self, num_classes = 18):
        super(Densenet121, self).__init__()
        self.model = models.densenet121(pretrained = True).features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(1024, out_features=num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.avgpool(x).view(x.size()[0], -1)
        x = self.fc(x)
        return x
    
    
class Densenet161(nn.Module):
    def __init__(self, num_classes = 18):
        super(Densenet161, self).__init__()
        self.model = models.densenet161(pretrained = True).features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(2208, out_features=num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.avgpool(x).view(x.size()[0], -1)
        x = self.fc(x)
        return x
    
    
class Densenet169(nn.Module):
    def __init__(self, num_classes = 18):
        super(Densenet169, self).__init__()
        self.model = models.densenet169(pretrained = True).features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(1664, out_features=num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.avgpool(x).view(x.size()[0], -1)
        x = self.fc(x)
        return x
    

class Swinnet(nn.Module):
    def __init__(self, num_classes = 18):
        super(Swinnet, self).__init__()
        self.swin = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        n_features = self.swin.head.in_features
        self.swin.head = nn.Linear(n_features, out_features=num_classes)

    def forward(self, x):
        x = self.swin(x)
        return x