import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm
from torchsummary import summary


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
    
class EfficientV2(nn.Module):
    def __init__(self, num_classes = 18):
        super(EfficientV2, self).__init__()
        self.v2 = timm.create_model('efficientnetv2_rw_t', pretrained= True)
        in_features = self.v2.classifier.in_features
        self.v2.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.v2(x)
        return x 

class EfficientB43way(nn.Module):
    def __init__(self, num_classes = 18):
        super(EfficientB43way, self).__init__()
        model = timm.create_model('efficientnet_b4', pretrained=True)

        self.model = nn.Sequential(*list(model.children())[:-1])

        self.mask = nn.Linear(1792, out_features=3)
        self.age = nn.Linear(1792, out_features=3)
        self.gender = nn.Linear(1792, out_features=2)

    def forward(self, x):
        x = self.model(x)
        
        mask = self.mask(x)
        age = self.age(x)
        gender = self.gender(x)

        return mask, gender, age
   
   
class Densenet2013way(nn.Module):
    def __init__(self, num_classes = 18):
        super(Densenet2013way, self ).__init__()
        self.model = torchvision.models.densenet201(pretrained = True).features

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
        self.mask = nn.Linear(1920, out_features=3)
        self.age = nn.Linear(1920, out_features=3)
        self.gender = nn.Linear(1920, out_features=2)


    def forward(self, x):
        x = self.model(x)
        x = self.avgpool(x).view(x.size()[0], -1)

        mask = self.mask(x)
        age = self.age(x)
        gender = self.gender(x)

        return mask, gender, age
    

class EfficientB43wayV2(nn.Module):
    def __init__(self, num_classes = 18):
        super(EfficientB43wayV2, self).__init__()
        model = timm.create_model('efficientnet_b4', pretrained=True)

        self.model = nn.Sequential(*list(model.children())[:-1])

        self.mask = nn.Linear(1792, out_features=3)
        self.age = nn.Linear(1792, out_features=3)
        self.gender = nn.Linear(1792, out_features=2)

    def forward(self, x):
        x = self.model(x)
        
        mask = self.mask(x)
        age = self.age(x)
        gender = self.gender(x)

        return mask, gender, age