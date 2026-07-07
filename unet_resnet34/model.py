"""
model.py
========
U-Net with a ResNet34 encoder for binary building segmentation.

Architecture
------------
Encoder  : ResNet34 (ImageNet pre-trained)
           Stages → (64, 64, 128, 256, 512) feature maps
Bridge   : ASPP (Atrous Spatial Pyramid Pooling) bottleneck
Decoder  : 4× up-sampling blocks with skip connections + residual conv
Head     : 1×1 conv → 1 channel (logit); sigmoid applied at inference

Usage
-----
    from model import UNetResNet34
    model = UNetResNet34(in_channels=3, num_classes=1, pretrained=True)
    logits = model(x)           # (B, 1, H, W), raw logits
    probs  = torch.sigmoid(logits)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights


# ──────────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────────

class ConvBnRelu(nn.Module):
    """Conv2d → BatchNorm → ReLU"""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 padding: int = 1, dilation: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """
    Bilinear up-sample + concatenate skip → two ConvBnRelu layers.
    A residual projection aligns channels for the identity shortcut.
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvBnRelu(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBnRelu(out_ch, out_ch)
        self.proj  = (
            nn.Conv2d(in_ch + skip_ch, out_ch, 1, bias=False)
            if (in_ch + skip_ch) != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x    = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                             align_corners=False)
        x    = torch.cat([x, skip], dim=1)
        res  = self.proj(x)
        out  = self.conv1(x)
        out  = self.conv2(out)
        return out + res   # residual shortcut


class ASPPModule(nn.Module):
    """
    Lightweight Atrous Spatial Pyramid Pooling bottleneck.
    Rates: 1, 6, 12, 18 + global average pooling branch.
    """
    def __init__(self, in_ch: int, out_ch: int = 256):
        super().__init__()
        self.b0 = ConvBnRelu(in_ch, out_ch, 1, padding=0)
        self.b1 = ConvBnRelu(in_ch, out_ch, 3, padding=6,  dilation=6)
        self.b2 = ConvBnRelu(in_ch, out_ch, 3, padding=12, dilation=12)
        self.b3 = ConvBnRelu(in_ch, out_ch, 3, padding=18, dilation=18)
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.project = ConvBnRelu(out_ch * 5, out_ch, 1, padding=0)
        self.drop    = nn.Dropout2d(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        gap  = F.interpolate(self.gap(x), size=(h, w), mode="bilinear",
                             align_corners=False)
        out  = torch.cat([self.b0(x), self.b1(x), self.b2(x),
                          self.b3(x), gap], dim=1)
        return self.drop(self.project(out))


# ──────────────────────────────────────────────────────────────────────────────
# Main model
# ──────────────────────────────────────────────────────────────────────────────

class UNetResNet34(nn.Module):
    """
    U-Net with ResNet34 encoder + ASPP bridge.

    Parameters
    ----------
    in_channels  : number of input channels (3 for RGB)
    num_classes  : 1 for binary segmentation
    pretrained   : load ImageNet weights for the encoder
    """

    # ResNet34 channel sizes per stage
    _ENC_CHANNELS = [64, 64, 128, 256, 512]

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        pretrained:  bool = True,
    ):
        super().__init__()

        # ── Encoder (ResNet34) ───────────────────────────────────────────────
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        base    = resnet34(weights=weights)

        if in_channels != 3:
            base.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2,
                                   padding=3, bias=False)

        self.enc0 = nn.Sequential(base.conv1, base.bn1, base.relu)  # /2  64ch
        self.pool = base.maxpool                                      # /4
        self.enc1 = base.layer1   # /4   64ch
        self.enc2 = base.layer2   # /8   128ch
        self.enc3 = base.layer3   # /16  256ch
        self.enc4 = base.layer4   # /32  512ch

        # ── Bridge (ASPP) ────────────────────────────────────────────────────
        self.bridge = ASPPModule(512, 256)

        # ── Decoder ──────────────────────────────────────────────────────────
        # Each block: (bridge/prev_out, skip_channels) → out_channels
        self.dec4 = DecoderBlock(256, 256, 256)   # skip from enc3
        self.dec3 = DecoderBlock(256, 128, 128)   # skip from enc2
        self.dec2 = DecoderBlock(128,  64, 64)    # skip from enc1
        self.dec1 = DecoderBlock( 64,  64, 64)    # skip from enc0

        # Final up-sample to original resolution (×2 because enc0 is /2)
        self.final_up = nn.Sequential(
            ConvBnRelu(64, 32),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBnRelu(32, 32),
        )

        self.head = nn.Conv2d(32, num_classes, 1)

        self._init_decoder_weights()

    # ── Weight init ──────────────────────────────────────────────────────────

    def _init_decoder_weights(self):
        for module in [self.bridge, self.dec4, self.dec3,
                       self.dec2, self.dec1, self.final_up, self.head]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                            nonlinearity="relu")
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e0 = self.enc0(x)           # (B,  64, H/2,  W/2)
        e1 = self.enc1(self.pool(e0))  # (B,  64, H/4,  W/4)
        e2 = self.enc2(e1)          # (B, 128, H/8,  W/8)
        e3 = self.enc3(e2)          # (B, 256, H/16, W/16)
        e4 = self.enc4(e3)          # (B, 512, H/32, W/32)

        # Bridge
        b  = self.bridge(e4)        # (B, 256, H/32, W/32)

        # Decoder
        d4 = self.dec4(b,  e3)      # (B, 256, H/16, W/16)
        d3 = self.dec3(d4, e2)      # (B, 128, H/8,  W/8)
        d2 = self.dec2(d3, e1)      # (B,  64, H/4,  W/4)
        d1 = self.dec1(d2, e0)      # (B,  64, H/2,  W/2)

        out = self.final_up(d1)     # (B,  32, H,    W)
        return self.head(out)       # (B,   1, H,    W)  – raw logits


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = UNetResNet34(pretrained=False)
    x = torch.randn(2, 3, 512, 512)
    y = model(x)
    print(f"Input : {x.shape}")
    print(f"Output: {y.shape}")   # should be (2, 1, 512, 512)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")
