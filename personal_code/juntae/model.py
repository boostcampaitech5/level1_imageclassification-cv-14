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
# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x

class CustomModel(nn.Module):
    def __init__(self, num_classes = 18):
        super().__init__()
        self.resnet = torchvision.models.resnet50(pretrained = True)
        self.resnet.fc = nn.Linear(2048 , num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x
    
class EfficientnetB4(nn.Module):
    def __init__(self, num_classes = 18):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = timm.create_model('efficientnet_b4', num_classes = self.num_classes, pretrained = True)

    def forward(self, x):
        x = self.backbone(x)
        return x

class EfficientnetB0(nn.Module):
    def __init__(self, num_classes = 18):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = timm.create_model('efficientnet_b0', num_classes = self.num_classes, pretrained = True)

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
    
class ConvnextSmall(nn.Module):
    def __init__(self, num_classes = 18):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = timm.create_model('convnext_small', num_classes = self.num_classes, pretrained = True)

    def forward(self, x):
        x = self.backbone(x)
        return x

class ConvnextTiny(nn.Module):
    def __init__(self, num_classes = 18):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = timm.create_model('convnext_tiny', num_classes = self.num_classes, pretrained = True)

    def forward(self, x):
        x = self.backbone(x)
        return x          