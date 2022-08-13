import torch.nn as nn
import torch
import clip
from torchvision.models import resnet50


# Overriding forward function
class RGB_ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = clip.load("RN50")[0].visual
        self.rgb_transform = nn.Linear(1024, 768)

    def forward(self, x):
        '''
            x : (B, L, 3, 224, 224)
        '''
        x = x.reshape(-1, 3, 224, 224)
        output = self.resnet(x).float()
        output = self.rgb_transform(output)

        return output

# Overriding forward function
class Depth_ResNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.resnet = resnet50(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(2048, config.hidden_size)

    def forward(self, x):
        '''
            x : (B, L, 3, 224, 224)
        '''
        
        x = x.reshape(-1, 1, 224, 224)
        output = self.resnet(x)

        return output