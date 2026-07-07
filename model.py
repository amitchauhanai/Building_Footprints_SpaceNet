"""
model.py  –  U-Net with ResNet-34 encoder for binary building segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet34_Weights


class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBnRelu(in_ch + skip_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UNetResNet34(nn.Module):
    def __init__(self, pretrained=True, num_classes=1, decoder_chs=(256, 128, 64, 32)):
        super().__init__()
        weights    = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        base       = models.resnet34(weights=weights)

        self.enc0  = nn.Sequential(base.conv1, base.bn1, base.relu)  # /2  64ch
        self.pool  = base.maxpool
        self.enc1  = base.layer1   # /4   64ch
        self.enc2  = base.layer2   # /8  128ch
        self.enc3  = base.layer3   # /16 256ch
        self.enc4  = base.layer4   # /32 512ch

        d0, d1, d2, d3 = decoder_chs
        self.dec4  = DecoderBlock(512, 256, d0)
        self.dec3  = DecoderBlock(d0,  128, d1)
        self.dec2  = DecoderBlock(d1,   64, d2)
        self.dec1  = DecoderBlock(d2,   64, d3)

        self.final_up = nn.Sequential(
            ConvBnRelu(d3, d3 // 2),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )
        self.head = nn.Conv2d(d3 // 2, num_classes, kernel_size=1)

        # initialise decoder
        for m in [self.dec4, self.dec3, self.dec2, self.dec1, self.final_up, self.head]:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.ones_(layer.weight); nn.init.zeros_(layer.bias)

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d  = self.dec4(e4, e3)
        d  = self.dec3(d,  e2)
        d  = self.dec2(d,  e1)
        d  = self.dec1(d,  e0)
        return self.head(self.final_up(d))   # raw logits (B,1,H,W)

    def freeze_encoder(self):
        for m in [self.enc0, self.pool, self.enc1, self.enc2, self.enc3, self.enc4]:
            for p in m.parameters(): p.requires_grad = False

    def unfreeze_encoder(self):
        for m in [self.enc0, self.pool, self.enc1, self.enc2, self.enc3, self.enc4]:
            for p in m.parameters(): p.requires_grad = True


def build_model(pretrained=True, device="cpu"):
    return UNetResNet34(pretrained=pretrained).to(device)


if __name__ == "__main__":
    m = build_model(pretrained=False)
    x = torch.randn(2, 3, 512, 512)
    print("out:", m(x).shape, " params:", sum(p.numel() for p in m.parameters()))
