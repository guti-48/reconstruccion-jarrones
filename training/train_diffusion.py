import sys
import os
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)
DATASET_DIR = os.path.join(BASE_DIR, 'data', 'processed')

from utils.visualize import mostrar_resultados
from torch.utils.data import DataLoader
from models.unet import UNet
from models.diffusion import add_noise
from utils.dataset import CeramicDataset



# Dataset
dataset = CeramicDataset(root_dir=DATASET_DIR)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Modelo
model = UNet(in_channels=3, out_channels=3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 5

for epoch in range(epochs):
    for masked, real, *_ in dataloader:

        t = torch.randint(1, 10, (1,))
        noisy, noise = add_noise(real, t)

        pred_noise = model(noisy)

        loss = ((pred_noise - noise) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} - Loss: {loss.item()}")
    
    # Extraemos el lote completo y nos quedamos solo con la posición 0 (masked) y 1 (real)
    batch = next(iter(dataloader))
    masked_test = batch[0][0:1]
    real_test = batch[1][0:1]
    
    mostrar_resultados(masked_test, real_test, model)