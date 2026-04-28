import os
import cv2
import torch
from torch.utils.data import DataLoader
from utils.dataset import CeramicDataset
from utils.models import EdgeModel

class Config:
    # Igual que la configuracion del train
    PATH = './checkpoints'
    DATA_ROOT = 'data/processed'
    GAN_LOSS = 'nsgan'
    LR = 0.0001
    D2G_LR = 0.1
    BETA1 = 0.0
    BETA2 = 0.9
    FM_LOSS_WEIGHT = 10
    BATCH_SIZE = 4
    EPOCHS = 5
    EDGE_THRESHOLD = 0.5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generar_bordes_para_M3():
    config = Config()    

    output_dir = 'data/results/edges'
    os.makedirs(output_dir, exist_ok=True)

    # Cargamos el dataset y el modelo
    dataset = CeramicDataset(root_dir=config.DATA_ROOT)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    edge_model = EdgeModel(config).to(config.DEVICE)
    edge_model.load()  # Carga el cerebro (.pth) de la carpeta checkpoints
    edge_model.eval()  # Ponemos la IA en "modo examen" (no aprende, solo escupe resultados)

    print(f"Generando bordes para {len(dataset)} imágenes...")
    
    # Bucle para generar y guardar foto a foto
    for idx, items in enumerate(loader):
        masked_rgb, img_rgb, img_gray, edges, masks = [item.to(config.DEVICE) for item in items]
        
        # El Falsificador se inventa las líneas del agujero
        with torch.no_grad():
            outputs = edge_model(img_gray, edges, masks)
        
        # MAGIA: Juntamos los bordes reales (fuera del agujero) con los inventados (dentro del agujero)
        outputs_merged = (outputs * masks) + (edges * (1 - masks))
        
        # Convertimos las matemáticas a una imagen física de píxeles (0 a 255)
        edge_img = outputs_merged.squeeze().cpu().numpy() * 255.0
        
        filename = dataset.files[idx]
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, edge_img)
        print(f" -> Guardado: {save_path}")

    print("\n¡Proceso terminado!")

if __name__ == "__main__":
    generar_bordes_para_M3()