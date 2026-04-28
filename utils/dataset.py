import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class CeramicDataset(Dataset):
    def __init__(self, root_dir):
        self.images_dir = os.path.join(root_dir, "images")
        self.masked_dir = os.path.join(root_dir, "masked_images")
        self.masks_dir = os.path.join(root_dir, "masks") 
        
        self.files = os.listdir(self.images_dir)

    def adaptive_canny(self, img_gray, sigma=0.33):
        # EL HACHAZO: Pintamos de negro puro el 30% inferior de la foto
        alto, ancho = img_gray.shape
        img_gray[int(alto * 0.70):alto, :] = 0
        
        # Filtro 3x3 para eliminar la textura del polvo y arañazos
        blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
        
        # Calcular la mediana solo de los píxeles válidos
        pixeles_validos = blurred[blurred > 0]
        v = np.median(pixeles_validos) if len(pixeles_validos) > 0 else 127
        
        # Generar bordes con los umbrales corregidos
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        return cv2.Canny(blurred, lower, upper)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]

        # Cargar las 3 imágenes
        img = cv2.imread(os.path.join(self.images_dir, filename))
        masked = cv2.imread(os.path.join(self.masked_dir, filename))
        mask = cv2.imread(os.path.join(self.masks_dir, filename), cv2.IMREAD_GRAYSCALE)

        # Redimensionar
        img = cv2.resize(img, (256, 256))
        masked = cv2.resize(masked, (256, 256))
        mask = cv2.resize(mask, (256, 256))
        
        # Procesar grises y bordes
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Llamamos al Canny mejorado
        edges = self.adaptive_canny(img_gray)
        
        # Convertir a tensores de 1 canal (1, 256, 256)
        edges_tensor = torch.tensor(edges).unsqueeze(0).float() / 255.0
        mask_tensor = torch.tensor(mask).unsqueeze(0).float() / 255.0
        img_gray_tensor = torch.tensor(img_gray).unsqueeze(0).float() / 255.0
        
        # Convertir a color y tensores de 3 canales (3, 256, 256)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        masked_rgb = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
        img_tensor = torch.tensor(img_rgb).permute(2, 0, 1).float() / 255.0
        masked_tensor = torch.tensor(masked_rgb).permute(2, 0, 1).float() / 255.0

        # Devolvemos siempre las 5 variables
        return masked_tensor, img_tensor, img_gray_tensor, edges_tensor, mask_tensor