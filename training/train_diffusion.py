import sys
import os
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from torch.utils.data import DataLoader
from models.unet import UNet
from models.diffusion import Diffusion
from utils.dataset import CeramicDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-4

dataset = CeramicDataset(root_dir='data/processed')
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

diffusion = Diffusion(T=200, device=device)  # puedes subir a 500 si quieres más calidad

model = UNet(in_channels=8, out_channels=3).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)


print("Entrenando Diffusion (fase final)...\n")

best_loss = float("inf")

for epoch in range(EPOCHS):

    total_loss = 0

    for masked, real, gray, edges, mask in dataloader:

        masked = masked.to(device)
        real = real.to(device)
        edges = edges.to(device)
        mask = mask.to(device)

        # =========================
        # SAMPLE t
        # =========================
        t = torch.randint(0, diffusion.T, (real.shape[0],), device=device).long()

        # =========================
        # FORWARD DIFFUSION
        # =========================
        x_t, noise = diffusion.add_noise(real, t)

        # =========================
        # INPUT FINAL
        # =========================
        input_model = torch.cat([x_t, masked, edges, mask], dim=1)

        # =========================
        # PREDICCIÓN
        # =========================
        predicted_noise = model(input_model, t)

        # =========================
        # LOSS (SOLO EN AGUJERO)
        # =========================
        loss = ((noise - predicted_noise) ** 2 * mask).sum() / (mask.sum() + 1e-8)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

    # =========================
    # GUARDAR MEJOR MODELO
    # =========================
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "checkpoints/diffusion_best.pth")
        print("Modelo guardado (mejor hasta ahora)")

print("\nEntrenamiento Diffusion completado")