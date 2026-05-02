import torch
import torch.nn as nn
import math

# =========================
# SINUSOIDAL TIME EMBEDDING
# =========================
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)

        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]

        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        return self.mlp(emb)


# =========================
# RESIDUAL BLOCK + GROUPNORM
# =========================


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()

        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.time_mlp = nn.Linear(time_dim, out_ch)

        self.act = nn.SiLU()  # mejor que ReLU

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        t = self.time_mlp(t_emb)[:, :, None, None]
        h = h + t

        h = self.act(self.norm2(h))
        h = self.conv2(h)

        return h + self.skip(x)


# =========================
# DOWN / UP BLOCKS
# =========================
class Down(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.block = ResBlock(in_ch, out_ch, time_dim)
        self.pool = nn.Conv2d(out_ch, out_ch, 4, 2, 1)  # mejor que maxpool

    def forward(self, x, t):
        x = self.block(x, t)
        return x, self.pool(x)


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_dim):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)

        # 👇 ahora usamos skip_ch correctamente
        self.block = ResBlock(out_ch + skip_ch, out_ch, time_dim)

    def forward(self, x, skip, t):
        x = self.up(x)

        if x.shape != skip.shape:
            x = nn.functional.interpolate(x, size=skip.shape[2:])

        x = torch.cat([x, skip], dim=1)
        x = self.block(x, t)
        return x


# =========================
# U-NET FINAL
# =========================
class UNet(nn.Module):
    def __init__(self, in_channels=8, out_channels=3, time_dim=256):
        super().__init__()

        self.time_embedding = TimeEmbedding(time_dim)

        # Encoder
        self.down1 = Down(in_channels, 64, time_dim)
        self.down2 = Down(64, 128, time_dim)
        self.down3 = Down(128, 256, time_dim)

        # Bottleneck
        self.mid = ResBlock(256, 256, time_dim)

        # Decoder
        self.up1 = Up(256, 256, 128, time_dim)
        self.up2 = Up(128, 128, 64, time_dim)
        self.up3 = Up(64, 64, 64, time_dim)

        # Output
        self.out = nn.Conv2d(64, out_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_embedding(t)

        # Encoder
        s1, x = self.down1(x, t_emb)
        s2, x = self.down2(x, t_emb)
        s3, x = self.down3(x, t_emb)

        # Bottleneck
        x = self.mid(x, t_emb)

        # Decoder
        x = self.up1(x, s3, t_emb)
        x = self.up2(x, s2, t_emb)
        x = self.up3(x, s1, t_emb)

        return self.out(x)