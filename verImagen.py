import torch, os, glob
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from utils.dataset import CeramicDataset
from models.unet import UNet
from models.diffusion import Diffusion

class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PATH = 'checkpoints' 
    ROOT_DIR = 'data/processed'
    EDGES_DIR = 'data/results/edges'
    #PATH = '/content/drive/MyDrive/Articulo-Investigacion/checkpoints' 
    
    # Rutas de datos (Igual que en train_main.py)
    #ROOT_DIR = '/content/data_rapida/processed'
    #EDGES_DIR = '/content/data_rapida/results/edges'

dispositivo = Config.DEVICE
config = Config()

print("Cargamos los datos y modelo")
dataset = CeramicDataset(root_dir=config.ROOT_DIR, edges_dir=config.EDGES_DIR)
loader = DataLoader(dataset, batch_size=1, shuffle=True)
items = next(iter(loader))

masked_tensor, img_tensor, _, edges_tensor, mask_tensor = [item.to(dispositivo) for item in items]

# Inicializamos a Oussama con sus 8 canales
modelo_difusion = UNet(in_channels=8, out_channels=3).to(dispositivo)

patron_busqueda = os.path.join(config.PATH, 'modelo_m3_diffusion_*.pth')
archivos_guardados = glob.glob(patron_busqueda)

if archivos_guardados:
    rutaConj = max(archivos_guardados, key=os.path.getmtime)
    print(f"   -> Encontrado checkpoint: {os.path.basename(rutaConj)}")
    
    # Cargamos el archivo físico
    checkpoint = torch.load(rutaConj, map_location=dispositivo)

    # Extracción segura de pesos
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        modelo_difusion.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        modelo_difusion.load_state_dict(checkpoint["state_dict"])
    else:
        modelo_difusion.load_state_dict(checkpoint)
        
    print(f"   -> [ÉXITO] Modelo M3 cargado y listo")
else:
    print(f"   -> [ERROR FATAL] No se encontró ningún archivo .pth en la carpeta '{config.PATH}'.")
    exit()

modelo_difusion.eval()
difusion_engine = Diffusion(T=200, device=dispositivo)

with torch.no_grad():
    print("Iniciando reconstrucción iterativa")
    reconstruida = difusion_engine.sample(modelo_difusion, masked_tensor, edges_tensor, mask_tensor)

print("Abriendo visor de imágenes...")
def to_numpy(x, is_gray=False):
    img = x[0].cpu().squeeze()
    if not is_gray:
        img = img.permute(1, 2, 0)
    return img.numpy().clip(0, 1)

# Visualización comparativa
plt.figure(figsize=(16, 5))
plt.subplot(1, 4, 1); plt.title("1. Plato Dañado"); plt.axis('off'); plt.imshow(to_numpy(masked_tensor))
plt.subplot(1, 4, 2); plt.title("2. Líneas GAN (M2)"); plt.axis('off'); plt.imshow(to_numpy(edges_tensor, True), cmap="gray")
plt.subplot(1, 4, 3); plt.title("3. Color Restaurado (M3)"); plt.axis('off'); plt.imshow(to_numpy(reconstruida))
plt.subplot(1, 4, 4); plt.title("4. Plato Original (Objetivo)"); plt.axis('off'); plt.imshow(to_numpy(img_tensor))

plt.tight_layout()
plt.show()