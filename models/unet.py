import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )

        self.pool = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        self.dec1 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1),
            nn.ReLU()
        )

        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))

        d1 = self.up(e2)
        d1 = torch.cat([d1, e1], dim=1)

        out = self.dec1(d1)
        return self.final(out)