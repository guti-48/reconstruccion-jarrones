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
    DATA_ROOT = '/content/data_rapida/processed'
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
    START_EPOCH = 51 
    GEN_WEIGHTS = 'EdgeModel_gen.pth' 
    DIS_WEIGHTS = 'EdgeModel_dis.pth' 

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

    # --- LÓGICA DE CARGA DE PESOS (RETOMAR ENTRENAMIENTO) ---
    if config.START_EPOCH > 1:
        print(f"Buscando pesos previos para retomar en la época {config.START_EPOCH}...")
        
        ruta_gen = os.path.join(config.PATH, config.GEN_WEIGHTS)
        ruta_dis = os.path.join(config.PATH, config.DIS_WEIGHTS)
        
        if os.path.exists(ruta_gen) and os.path.exists(ruta_dis):
            # 1. Cargamos los "maletines" enteros a la memoria
            checkpoint_gen = torch.load(ruta_gen, map_location=config.DEVICE)
            checkpoint_dis = torch.load(ruta_dis, map_location=config.DEVICE)
            
            # 2. Extraemos SOLO los pesos reales ignorando el número de iteración
            pesos_gen = checkpoint_gen['generator'] if 'generator' in checkpoint_gen else checkpoint_gen
            
            # (Hacemos lo mismo para el discriminador, asumiendo que su clave se llama así)
            pesos_dis = checkpoint_dis['discriminator'] if 'discriminator' in checkpoint_dis else checkpoint_dis

            # 3. Le inyectamos los pesos limpios a los modelos
            edge_model.generator.load_state_dict(pesos_gen)
            edge_model.discriminator.load_state_dict(pesos_dis)
            
            print("¡Memoria antigua inyectada con éxito! La IA recuerda su entrenamiento.")
        else:
            print("⚠️ ADVERTENCIA: No se encontraron los archivos .pth en la carpeta checkpoints.")
            print("Asegúrate de que los nombres GEN_WEIGHTS y DIS_WEIGHTS son correctos.")
            print("El entrenamiento va a empezar desde cero por seguridad.")
            config.START_EPOCH = 1
    else:
        print("Empezando entrenamiento desde cero...")

    # Bucle de Entrenamiento (Modificado para usar START_EPOCH)
    for epoch in range(config.START_EPOCH, config.EPOCHS + 1):
        print(f"\nÉpoca {epoch}/{config.EPOCHS}")
        edge_model.train()
        
        epoch_gen_loss = 0
        epoch_dis_loss = 0
        epoch_precision = 0

        for batch_idx, items in enumerate(train_loader):
            masked_rgb, img_rgb, img_gray, edges, masks = [item.to(config.DEVICE) for item in items]

            outputs, gen_loss, dis_loss, logs = edge_model.process(img_gray, edges, masks)

            precision, recall = edgeacc(edges * masks, outputs * masks)

            edge_model.backward(gen_loss, dis_loss)

            epoch_gen_loss += gen_loss.item()
            epoch_dis_loss += dis_loss.item()
            epoch_precision += precision.item()

        num_batches = len(train_loader)
        print(f" -> Pérdida Falsificador (Gen): {epoch_gen_loss/num_batches:.4f}")
        print(f" -> Pérdida Policía (Dis): {epoch_dis_loss/num_batches:.4f}")
        print(f" -> Precisión de Bordes: {(epoch_precision/num_batches)*100:.2f}%")
        
        if epoch % 10 == 0:
            # Tu método save() de EdgeModel ya sobrescribe los archivos con el progreso actual
            edge_model.save()
            print(f"--> Punto de control guardado en la época {epoch}")

    edge_model.save()
    print("\n¡Entrenamiento GAN completado!")

if __name__ == "__main__":
    train_gan()