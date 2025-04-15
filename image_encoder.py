import torch.nn as nn
import torchvision.models as models

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # Remove final layers

    def forward(self, x):
        return self.features(x)  # shape: (batch, 512, H/32, W/32)
