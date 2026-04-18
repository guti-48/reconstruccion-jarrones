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
    
    # RECIBIMOS LAS 5 VARIABLES OFICIALES DEL DATASET
    masked_tensor, img_tensor, img_gray_tensor, edges_tensor, mask_tensor = dataset[0]
    
    # Convertimos los tensores matemáticos de PyTorch a imágenes de Numpy
    # permute(1,2,0) cambia de (Canales, Alto, Ancho) a (Alto, Ancho, Canales)
    masked_np = masked_tensor.permute(1, 2, 0).numpy()
    img_np = img_tensor.permute(1, 2, 0).numpy()
    
    # squeeze() elimina la dimensión del canal único para que quede (256, 256) puro en blanco y negro
    img_gray_np = img_gray_tensor.squeeze().numpy()
    edges_np = edges_tensor.squeeze().numpy() 
    mask_np = mask_tensor.squeeze().numpy()
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    axes[0].imshow(img_np)
    axes[0].set_title('1. Ground Truth (Original RGB)')
    axes[0].axis('off')
    
    axes[1].imshow(masked_np)
    axes[1].set_title('2. Imagen Dañada')
    axes[1].axis('off')

    axes[2].imshow(img_gray_np, cmap='gray')
    axes[2].set_title('3. Escala de Grises')
    axes[2].axis('off')
    
    axes[3].imshow(edges_np, cmap='gray')
    axes[3].set_title('4. Boceto Canny')
    axes[3].axis('off')

    axes[4].imshow(mask_np, cmap='gray')
    axes[4].set_title('5. Máscara')
    axes[4].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_dataset()