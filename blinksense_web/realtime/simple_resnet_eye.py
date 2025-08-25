import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """
    ResNet 'basic' block: 3x3 -> 3x3 with a skip connection.
    If spatial size or channels change, a 1x1 projection aligns the skip.
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            identity = self.down(identity)
        out = F.relu(out + identity, inplace=True)
        return out
    
# A tiny ResNet-style CNN for 64x64 grayscale eye crops
class SimpleResNetEye(nn.Module):
    """
    Stages (input 1x64x64):
      Stem:   3x3 conv -> BN -> ReLU -> MaxPool(2)    # 64 -> 32
      L1:     2 x BasicBlock(32)                      # 32 -> 32
      L2:     BasicBlock(32->64, stride=2) + Block    # 32 -> 16
      L3:     BasicBlock(64->96, stride=2) + Block    # 16 -> 8
      GAP:    AdaptiveAvgPool -> 96
      Head:   96 -> emb(64) -> logits(2)
    """
    def __init__(self, emb_dim=64, num_classes=2):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 64x64 -> 32x32
        )
        # Residual stages
        self.layer1 = nn.Sequential(
            BasicBlock(32, 32, stride=1),
            BasicBlock(32, 32, stride=1),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(32, 64, stride=2),  # 32x32 -> 16x16
            BasicBlock(64, 64, stride=1),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(64, 96, stride=2),  # 16x16 -> 8x8
            BasicBlock(96, 96, stride=1),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)   # -> [B, 96, 1, 1]
        self.fc  = nn.Sequential(
            nn.Linear(96, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
        self.head = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        x = self.stem(x)     # [B, 32, 32, 32]
        x = self.layer1(x)   # [B, 32, 32, 32]
        x = self.layer2(x)   # [B, 64, 16, 16]
        x = self.layer3(x)   # [B, 96,  8,  8]
        x = self.gap(x).flatten(1)  # [B, 96]
        z = self.fc(x)       # [B, emb_dim]
        logits = self.head(z)  # [B, 2]
        return logits, z