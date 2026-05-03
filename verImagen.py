import torch
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from utils.dataset import CeramicDataset
from utils.models import EdgeModel
from models.unet import UNet
from models.diffusion import Diffusion

# Configuración mínima
class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PATH = 'checkpoints'
    GAN_LOSS = 'nsgan'
    LR = 1e-4
    BETA1 = 0.0
    BETA2 = 0.999
    D2G_LR = 0.1
    FM_LOSS_WEIGHT = 10.0

dispositivo = Config.DEVICE

print("Imagen aleatoria")
dataset = CeramicDataset(root_dir='data/processed')
loader = DataLoader(dataset, batch_size=1, shuffle=True)
items = next(iter(loader))
masked_tensor, img_tensor, img_gray_tensor, edges_tensor, mask_tensor = [item.to(dispositivo) for item in items]

print("Cargamos modelos")
# Cargamos a Rafa (Sabe dibujar)
config = Config()
modelo_bordes = EdgeModel(config).to(dispositivo)
modelo_bordes.load()
modelo_bordes.eval()

# Cargamos a Oussama (Sabe pintar)
modelo_difusion = UNet(in_channels=8, out_channels=3).to(dispositivo)

ruta_oussama = os.path.join("checkpoints", "modeloConjunto_epoch8_ssim0.0066.pth")

if os.path.exists(ruta_oussama):
    # Cargamos el archivo físico
    checkpoint = torch.load(ruta_oussama, map_location=dispositivo)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        modelo_difusion.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        modelo_difusion.load_state_dict(checkpoint["state_dict"])
    else:
        modelo_difusion.load_state_dict(checkpoint)
        
    print(f"   -> ¡Pesos de M3 encontrados y cargados! ({ruta_oussama})")
else:
    print(f"   -> [ERROR FATAL] No se encontró el archivo: {ruta_oussama}")

modelo_difusion.eval()
difusion_engine = Diffusion(T=200, device=dispositivo)

with torch.no_grad():
    boceto, _, _, _ = modelo_bordes.process(img_gray_tensor, edges_tensor, mask_tensor)
    
    reconstruida = difusion_engine.sample(modelo_difusion, masked_tensor, boceto, mask_tensor)

print("Abriendo visor de imágenes...")
def to_numpy(x, is_gray=False):
    img = x[0].cpu().squeeze()
    if not is_gray:
        img = img.permute(1, 2, 0)
    return img.numpy().clip(0, 1)

plt.figure(figsize=(16, 5))
plt.subplot(1, 4, 1); plt.title("1. Plato Dañado"); plt.axis('off'); plt.imshow(to_numpy(masked_tensor))
plt.subplot(1, 4, 2); plt.title("2. Líneas Inventadas (M2)"); plt.axis('off'); plt.imshow(to_numpy(boceto, True), cmap="gray")
plt.subplot(1, 4, 3); plt.title("3. Color Restaurado (M3)"); plt.axis('off'); plt.imshow(to_numpy(reconstruida))
plt.subplot(1, 4, 4); plt.title("4. Plato Original (Objetivo)"); plt.axis('off'); plt.imshow(to_numpy(img_tensor))

plt.tight_layout()
plt.show()