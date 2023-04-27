import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm
import torch
import os

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
class EfficientnetB4(nn.Module):
    def __init__(self, num_classes = 18):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = timm.create_model('efficientnet_b4', num_classes = self.num_classes, pretrained = True)

    def forward(self, x):
        x = self.backbone(x)
        return x
    

class EfficientnetB43way(nn.Module):
    def __init__(self, num_classes = 18):
        super(EfficientnetB43way, self).__init__()
        model = timm.create_model('efficientnet_b4', pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-1])

        self.mask = nn.Sequential(
            nn.Linear(1792, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 3),
        )

        self.age = nn.Sequential(
            nn.Linear(1792, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 3),
        ) 

        self.gender = nn.Sequential(
            nn.Linear(1792, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 2),
        )  

    def forward(self, x):
        x = self.model(x)
        
        mask = self.mask(x)
        age = self.age(x)
        gender = self.gender(x)

        return mask, gender, age

class EfficientnetB43wayF(nn.Module):
    def __init__(self, num_classes = 18):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = EfficientnetB4()

        best_model_path = ''
        self.backbone.load_state_dict(torch.load(best_model_path))
        
        for name, param in self.backbone.backbone.named_parameters():
            param.requires_grad = False

        self.backbone.backbone.classifier = nn.Linear(1792, 512)

        self.mask = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 3),
        )

        self.age = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 3),
        ) 

        self.gender = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 2),
        )  

    def forward(self, x):
        x = self.backbone(x)
        mask = self.mask(x)
        age = self.age(x)
        gender = self.gender(x)

        return mask, gender, age
    
