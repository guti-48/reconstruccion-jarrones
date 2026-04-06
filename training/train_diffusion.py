import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from utils.visualize import mostrar_resultados
from torch.utils.data import DataLoader
from models.unet import UNet
from models.diffusion import add_noise
from utils.dataset import CeramicDataset

# Dataset
dataset = CeramicDataset("../imagenes_procesadas")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Modelo
model = UNet(in_channels=3, out_channels=3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 5

for epoch in range(epochs):
    for masked, real in dataloader:

        t = torch.randint(1, 10, (1,))
        noisy, noise = add_noise(real, t)

        pred_noise = model(noisy)

        loss = ((pred_noise - noise) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} - Loss: {loss.item()}")
    masked, real = next(iter(dataloader))
    mostrar_resultados(masked[0:1], real[0:1], model)


