import matplotlib.pyplot as plt
import sys
import os


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)
DATASET_DIR = os.path.join(BASE_DIR, 'data', 'processed')

from utils.dataset import *


def test_dataset():
    
    dataset = CeramicDataset(root_dir=DATASET_DIR)
    
    print(f"Total de imágenes en el dataset: {len(dataset)}")
    
    # 1. RECIBIMOS LAS 3 VARIABLES OFICIALES DEL DATASET
    masked_tensor, img_tensor, edges_tensor = dataset[0]
    
    # 2. Convertimos los tensores matemáticos de PyTorch a imágenes de Numpy
    # permute(1,2,0) cambia de (Canales, Alto, Ancho) a (Alto, Ancho, Canales)
    masked_np = masked_tensor.permute(1, 2, 0).numpy()
    img_np = img_tensor.permute(1, 2, 0).numpy()
    
    # squeeze() elimina la dimensión del canal único para que quede (256, 256) puro en blanco y negro
    edges_np = edges_tensor.squeeze().numpy() 
    
    # 3. Dibujamos 3 columnas
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_np)
    axes[0].set_title('1. Ground Truth (Original RGB)')
    axes[0].axis('off')
    
    axes[1].imshow(masked_np)
    axes[1].set_title('2. Imagen Dañada (Input para M3)')
    axes[1].axis('off')
    
    axes[2].imshow(edges_np, cmap='gray')
    axes[2].set_title('3. Boceto Canny (Input para M2)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_dataset()