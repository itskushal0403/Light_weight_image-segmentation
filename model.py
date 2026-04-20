import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class LiteSeg(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()

        # Pretrained MobileNetV2 backbone
        backbone = models.mobilenet_v2(weights="DEFAULT")

        # Take early layers → low FLOPs
        self.features = backbone.features[:7]  # output channels = 32

        # Decoder (very light)
        self.conv1 = nn.Conv2d(32, 24, kernel_size=1)
        self.conv2 = nn.Conv2d(24, 16, kernel_size=1)
        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):

        x = self.features(x)  # Downsampled features

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv1(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv2(x)

        x = F.interpolate(x, size=(300, 300), mode='bilinear', align_corners=False)
        x = self.final(x)

        return x