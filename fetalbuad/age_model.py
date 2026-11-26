"""Multitask 3D network for global age, tissue segmentation and voxel age."""

import torch
from torch import nn
from torch.nn import functional as F


def _conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(),
        nn.Conv3d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(),
    )


class VoxelAgeNet(nn.Module):
    """Predict age in days and logits for nine tissue classes."""

    def __init__(self):
        super().__init__()
        self.encoder1 = _conv_block(1, 32)
        self.encoder2 = _conv_block(32, 64)
        self.encoder3 = _conv_block(64, 128)
        self.encoder4 = _conv_block(128, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.upsample3 = nn.ConvTranspose3d(256, 256, 2, stride=2)
        self.decoder3 = _conv_block(256 + 128, 128)
        self.upsample2 = nn.ConvTranspose3d(128, 128, 2, stride=2)
        self.decoder2 = _conv_block(128 + 64, 64)
        self.upsample1 = nn.ConvTranspose3d(64, 64, 2, stride=2)
        self.decoder1 = _conv_block(64 + 32, 32)
        self.outputconv = nn.Conv3d(32, 9, 1)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(256, 1)
        self.outputconv_age = nn.Conv3d(32, 1, 1)

    def forward(self, image):
        first = self.encoder1(image)
        second = self.encoder2(F.max_pool3d(first, 2))
        third = self.encoder3(F.max_pool3d(second, 2))
        latent = self.dropout1(self.encoder4(F.max_pool3d(third, 2)))
        global_age = self.fc(self.dropout(self.avgpool(latent).flatten(1)))
        decoded = self.decoder3(torch.cat((self.upsample3(latent), third), dim=1))
        decoded = self.decoder2(torch.cat((self.upsample2(decoded), second), dim=1))
        decoded = self.decoder1(torch.cat((self.upsample1(decoded), first), dim=1))
        return global_age, self.outputconv(decoded), self.outputconv_age(decoded)


def load_age_model(checkpoint, device="cpu"):
    """Load a frozen brain-age network for paired MRI and reconstruction inference."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = VoxelAgeNet()
    model.load_state_dict(payload.get("net", payload), strict=True)
    return model.to(device, memory_format=torch.channels_last_3d).eval().requires_grad_(False)
