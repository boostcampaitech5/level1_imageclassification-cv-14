import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm

class Densenet201(nn.Module):
        def __init__(self, num_classes = 18):
            super(Densenet201, self).__init__()
            self.model = torchvision.models.densenet201(pretrained = True).features
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(1920, num_classes)

        def forward(self, x):
            x = self.model(x)
            x = self.avgpool(x).view(x.size()[0], -1)
            x = self.fc(x)
            return x

class EfficientB4(nn.Module):
    def __init__(self, num_classes = 18):
        super(EfficientB4, self).__init__()
        self.effnet = timm.create_model('efficientnet_b4', pretrained=True)
        in_features = self.effnet.classifier.in_features
        self.effnet.classifier = nn.Linear(in_features , num_classes)

    def forward(self, x):
        x = self.effnet(x)
        return x
    
    
class AdvProp(nn.Module):
    def __init__(self, num_classes = 18):
        super(AdvProp, self).__init__()
        self.advprop = timm.create_model('tf_efficientnet_b4_ap', pretrained=True)
        in_features = self.advprop.classifier.in_features
        self.advprop.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
         x = self.advprop(x)
         return x 
    

