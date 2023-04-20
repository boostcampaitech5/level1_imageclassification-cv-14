import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm


class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# --EffecientNet B0 pretrained=True

class EfficientnetB0(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True)
        self.backbone.classifier = nn.Linear(1280, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        return x


# --EffecientNet B1 pretrained=True

class EfficientnetB1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.backbone = timm.create_model('efficientnet_b1', pretrained=True)
        self.backbone.classifier = nn.Linear(1280, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        return x
    
# --EffecientNet B2 pretrained=True

class EfficientnetB2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.backbone = timm.create_model('efficientnet_b2', pretrained=True)
        self.backbone.classifier = nn.Linear(1408, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        return x
    

# --EffecientNet B3 pretrained=True

class EfficientnetB3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.backbone = timm.create_model('efficientnet_b3', pretrained=True)
        self.backbone.classifier = nn.Linear(1536, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        return x

# --EffecientNet B4 pretrained=True

class EfficientnetB4(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.num_classes = num_classes
        self.backbone = timm.create_model('efficientnet_b4', num_classes = self.num_classes, pretrained = True)
        
        # self.backbone = timm.create_model('efficientnet_b4', pretrained=True)
        # self.backbone.classifier = nn.Linear(1792, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        return x
    

# --Vit_small_patch16_224 pretrained=True

class Vitsmall_patch16_224(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.backbone = timm.create_model('vit_small_patch16_224', pretrained=True)
        self.backbone.head = nn.Linear(384, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        return x
    

# --Resnet34 pretrained=True 

class Resnet34(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.backbone = models.resnet34(pretrained=True)
        self.backbone.head = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        return x
    

# --Resnet50 pretrained=True 

class Resnet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.head = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        return x