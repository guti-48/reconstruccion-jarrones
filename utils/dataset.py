import os, cv2, torch, glob
import numpy as np
from torch.utils.data import Dataset

class CeramicDataset(Dataset):
    def __init__(self, root_dir, edges_dir=None):
        self.images_dir = os.path.join(root_dir, "images")
        self.masked_dir = os.path.join(root_dir, "masked_images")
        self.masks_dir = os.path.join(root_dir, "masks") 
        if edges_dir is None:
            # Asume que root_dir es 'data/processed', subimos un nivel para ir a 'results/edges'
            base_data_dir = os.path.dirname(root_dir)
            self.edges_dir = os.path.join(base_data_dir, "results", "edges")
        else:
            self.edges_dir = edges_dir
            
        self.files = os.listdir(self.images_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        
        # 1. Carga segura de imágenes originales y máscaras
        img = cv2.imread(os.path.join(self.images_dir, filename))
        masked = cv2.imread(os.path.join(self.masked_dir, filename))
        mask = cv2.imread(os.path.join(self.masks_dir, filename), cv2.IMREAD_GRAYSCALE)

        # 2. Búsqueda robusta del borde GAN (Ignora la extensión y mayúsculas)
        base_name = os.path.splitext(filename)[0] 
        
        patron_busqueda = os.path.join(self.edges_dir, f"esqueleto_{base_name}.*")
        archivos_encontrados = glob.glob(patron_busqueda)

        if not archivos_encontrados:
            raise FileNotFoundError(f"[ERROR CRÍTICO] Falta el borde de {base_name} en la carpeta edges. ¿Corriste genera_bordes.py para esta imagen?")
        
        gan_edge_path = archivos_encontrados[0] 
        edges = cv2.imread(gan_edge_path, cv2.IMREAD_GRAYSCALE)

        # 3. Redimensionar
        img = cv2.resize(img, (256, 256))
        masked = cv2.resize(masked, (256, 256))
        mask = cv2.resize(mask, (256, 256))
        edges = cv2.resize(edges, (256, 256))
        
        # 4. Tensores [1, 256, 256]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray_tensor = torch.tensor(img_gray).unsqueeze(0).float() / 255.0
        mask_tensor = torch.tensor(mask).unsqueeze(0).float() / 255.0
        edges_tensor = torch.tensor(edges).unsqueeze(0).float() / 255.0
        
        # 5. Tensores RGB [3, 256, 256]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        masked_rgb = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
        img_tensor = torch.tensor(img_rgb).permute(2, 0, 1).float() / 255.0
        masked_tensor = torch.tensor(masked_rgb).permute(2, 0, 1).float() / 255.0
        # 1. Creamos una silueta: Todo lo que sea más brillante que 0.1 (casi negro) es plato (1). El fondo es (0).
        silueta_plato = (img_gray_tensor > 0.1).float()

        # 2. Multiplicamos la máscara de daño por la silueta. 
        # Si un trazo blanco de daño sale al fondo negro (0), se convierte en 0 (se elimina).
        mask_tensor = mask_tensor * silueta_plato

        return masked_tensor, img_tensor, img_gray_tensor, edges_tensor, mask_tensor