import sys
import os
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)
DATASET_DIR = os.path.join(BASE_DIR, 'data', 'processed')
from torch.utils.data import DataLoader

# M1
from utils.dataset import CeramicDataset
from utils.checkpoint import CheckpointManager

# M2 (GAN - Bordes)
from models.networks import EdgeGenerator

# M3 (Diffusion - Textura)
from models.unet import UNet
from models.diffusion import Diffusion

# =========================
# CONFIG
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
diffusion=Diffusion(device=device)
BATCH_SIZE = 4
EPOCHS = 20
LR = 1e-4

# =========================
# DATASET
# =========================
dataset = CeramicDataset(root_dir='data/processed')
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# =========================
# MODELOS
# =========================

# 🔹 GAN (solo generador para inferencia)
edge_generator = EdgeGenerator().to(device)

# ⚠️ Cargar pesos entrenados del GAN
edge_model = EdgeGenerator().to(device)

checkpoint = torch.load("checkpoints/edgeModel_gen.pth", map_location=device)
edge_model.load_state_dict(checkpoint["generator"])
edge_generator.eval()  # MUY IMPORTANTE

# 🔹 Diffusion (U-Net guiado)
unet = UNet(in_channels=8, out_channels=3).to(device)

# 🔥 CARGAR MODELO PREENTRENADO
unet.load_state_dict(torch.load("checkpoints/diffusion_best.pth", map_location=device))
print("✔ Diffusion preentrenado cargado")

optimizer = torch.optim.Adam(unet.parameters(), lr=LR)

# =========================
# CHECKPOINT
# =========================
guardian = CheckpointManager(save_dir='checkpoints', model_name='modeloConjunto')

# =========================
# TRAIN LOOP
# =========================
print(" Iniciando entrenamiento conjunto GAN + Diffusion...\n")

print(" Entrenamiento conjunto...\n")



best_loss = float("inf")

for epoch in range(EPOCHS):

    total_loss = 0

    for masked, real, gray, edges, mask in dataloader:

        masked = masked.to(device)
        real = real.to(device)
        edges = edges.to(device)
        mask = mask.to(device)

        # ✅ YA USAMOS EDGES DEL DATASET
        fake_edges = edges

        # ===== SAMPLE t =====
        t = torch.randint(0, diffusion.T, (real.shape[0],), device=device).long()
        

        # ===== FORWARD DIFFUSION =====
        x_t, noise = diffusion.add_noise(real, t)

        # ===== INPUT =====
        input_model = torch.cat([x_t, masked, fake_edges, mask], dim=1)

        # ===== PRED =====
        predicted_noise = unet(input_model, t)

        # ===== LOSS =====
        loss = ((noise - predicted_noise) ** 2 * mask).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")

    # ✅ GUARDAR MEJOR
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(unet.state_dict(), "checkpoints/diffusion_best.pth")
        print("✔ Modelo mejor guardado")

    # ✅ VISUALIZACIÓN CONTROLADA
    if epoch % 5 == 0:
        from utils.visualize import mostrar_resultados

        batch = next(iter(dataloader))
        masked_test, real_test, gray_test, edges_test, mask_test = batch

        mostrar_resultados(masked_test, real_test, edges_test, mask_test, unet, diffusion)
print("\n✅ Entrenamiento conjunto completado")