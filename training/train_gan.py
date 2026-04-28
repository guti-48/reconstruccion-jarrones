import os
import sys
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset import CeramicDataset
from utils.models import EdgeModel
from utils.metrics import EdgeAccuracy

class Config:
    # Parámetros del entrenamiento
    PATH = './checkpoints'
    DATA_ROOT = 'data/processed'
    GAN_LOSS = 'nsgan'
    LR = 0.0001
    D2G_LR = 0.1
    BETA1 = 0.0
    BETA2 = 0.9
    FM_LOSS_WEIGHT = 10
    BATCH_SIZE = 4
    EPOCHS = 150
    EDGE_THRESHOLD = 0.5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_gan():
    config = Config()
    os.makedirs(config.PATH, exist_ok=True)

    print("--- Iniciando Entrenamiento de Patrones (Edge GAN) ---")
    print(f"Usando dispositivo: {config.DEVICE}")

    # Cargar el Dataset y Modelo
    train_dataset = CeramicDataset(root_dir=config.DATA_ROOT)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
    
    edge_model = EdgeModel(config).to(config.DEVICE)
    edgeacc = EdgeAccuracy(config.EDGE_THRESHOLD).to(config.DEVICE)

    # Bloque de carga eliminado para empezar desde cero
    print("Empezando entrenamiento...")

    # Bucle de Entrenamiento
    for epoch in range(1, config.EPOCHS + 1):
        print(f"\nÉpoca {epoch}/{config.EPOCHS}")
        edge_model.train()
        
        epoch_gen_loss = 0
        epoch_dis_loss = 0
        epoch_precision = 0

        for batch_idx, items in enumerate(train_loader):
            # Le pasa las 5 variables que devuelve el Dataset a la GPU/CPU
            masked_rgb, img_rgb, img_gray, edges, masks = [item.to(config.DEVICE) for item in items]

            # Entrena el modelo con las imágenes en gris, los bordes Canny y la máscara
            outputs, gen_loss, dis_loss, logs = edge_model.process(img_gray, edges, masks)

            # Calcula la precisión
            precision, recall = edgeacc(edges * masks, outputs * masks)

            # Actualiza pesos (Backpropagation)
            edge_model.backward(gen_loss, dis_loss)

            # Acumula las métricas
            epoch_gen_loss += gen_loss.item()
            epoch_dis_loss += dis_loss.item()
            epoch_precision += precision.item()

        num_batches = len(train_loader)
        print(f" -> Pérdida Falsificador (Gen): {epoch_gen_loss/num_batches:.4f}")
        print(f" -> Pérdida Policía (Dis): {epoch_dis_loss/num_batches:.4f}")
        print(f" -> Precisión de Bordes: {(epoch_precision/num_batches)*100:.2f}%")
        
        if epoch % 10 == 0:
            edge_model.save()
            print(f"--> Punto de control guardado en la época {epoch}")

    edge_model.save()
    print("\n¡Entrenamiento GAN completado!")

if __name__ == "__main__":
    train_gan()