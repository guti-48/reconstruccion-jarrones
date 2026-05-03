import torch, os, time, glob
import torch.nn as nn
from torch.utils.data import DataLoader

# Importaciones M1 (guti)
from utils.dataset import CeramicDataset
from utils.metrics import calcular_metricas, EdgeAccuracy
from utils.checkpoint import CheckpointManager

# Importaciones M2 (rafa)
from utils.models import EdgeModel

# Importaciones M3 (oussama)
from models.unet import UNet
from models.diffusion import Diffusion # [M1 FIX] Importamos su nueva clase orientada a objetos

# config global
dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Usando dispositivo: {dispositivo}")
if dispositivo.type == 'cpu':
    print("[AVISO] Entrenando en CPU. Cada época puede tardar 30-60 min.")

class Config:
    DEVICE         = dispositivo
    PATH           = 'checkpoints'
    GAN_LOSS       = 'nsgan'
    LR             = 1e-4
    BETA1          = 0.0
    BETA2          = 0.999
    D2G_LR         = 0.1
    FM_LOSS_WEIGHT = 10.0
    EDGE_THRESHOLD = 0.5
    BATCH_SIZE     = 4
    EPOCHS         = 100
    NUM_WORKERS    = 0 if os.name == 'nt' else 4
    LOG_INTERVAL   = 10
    DIFFUSION_T    = 200 

config = Config()

# input de datos a usar
dataset = CeramicDataset(root_dir='data/processed')
cargaDatos = DataLoader(
    dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=config.NUM_WORKERS,
    pin_memory=(dispositivo.type == 'cuda')
)
print(f"[INFO] Dataset: {len(dataset)} imágenes | {len(cargaDatos)} batches por época")

# Checkpoint que nos guarda el conjunto mejor
guardian = CheckpointManager(save_dir='checkpoints', model_name='modeloConjunto')

# MODELOS
print("Cargando EdgeModel (M2) y UNet de Difusión (M3)...")

# Inicializamos modelo de Rafa y congelamos
modelo_bordes = EdgeModel(config).to(dispositivo)
modelo_bordes.load()  
modelo_bordes.eval() # Congelado para que M3 aprenda más rápido

# Inicislizamos modelo de Oussama con 8 canales
modelo_difusion = UNet(in_channels=8, out_channels=3).to(dispositivo)
difusion_engine = Diffusion(T=config.DIFFUSION_T, device=dispositivo)

patron_busqueda = os.path.join(config.PATH, 'modeloConjunto_epoch*.pth')
archivos_guardados = glob.glob(patron_busqueda)

if archivos_guardados:
    # Si encuentra archivos, cogemos el que se modificó más recientemente
    ruta_m3 = max(archivos_guardados, key=os.path.getmtime)
    
    checkpoint = torch.load(ruta_m3, map_location=dispositivo)
    # Extracción segura de pesos
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        modelo_difusion.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        modelo_difusion.load_state_dict(checkpoint["state_dict"])
    else:
        modelo_difusion.load_state_dict(checkpoint)
        
    nombre_archivo = os.path.basename(ruta_m3)
    print(f"  -> [ÉXITO] Se retoma M3 desde el archivo automático: {nombre_archivo}")
else:
    print("  -> [AVISO] No hay checkpoints previos. Empezando Oussama (M3) desde cero.")

optimizador_difusion = torch.optim.Adam(modelo_difusion.parameters(), lr=config.LR)
edgeacc = EdgeAccuracy(threshold=config.EDGE_THRESHOLD)

# Precisión mixta
use_amp = (dispositivo.type == 'cuda')
scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)


# bucle para el entrenamiento conjunyo
for epoch in range(1, config.EPOCHS + 1):
    print(f"\n{'='*50}")
    print(f"  Época {epoch}/{config.EPOCHS}")
    print(f"{'='*50}")

    modelo_difusion.train()

    epoch_gen_loss    = 0.0
    epoch_dis_loss    = 0.0
    epoch_precision   = 0.0
    loss_m3_acumulada = 0.0
    t_inicio_epoca    = time.time()

    for batch_idx, items in enumerate(cargaDatos):
        masked_tensor, img_tensor, img_gray_tensor, edges_tensor, mask_tensor = [
            item.to(dispositivo) for item in items
        ]

        # FASE 1: INFERENCIA GAN DE BORDES (Rafa)
        with torch.no_grad(): 
            with torch.cuda.amp.autocast(enabled=use_amp):
                boceto_predicho, gen_loss, dis_loss, logs_m2 = modelo_bordes.process(
                    img_gray_tensor, edges_tensor, mask_tensor
                )

        precision, recall = edgeacc(
            edges_tensor * mask_tensor,
            boceto_predicho * mask_tensor
        )

        # FASE 2: ENTRENAMIENTO DIFUSIÓN (Oussama)
        t = torch.randint(0, difusion_engine.T, (img_tensor.shape[0],), device=dispositivo).long()
        
        # Inyectar ruido
        noisy_rgb, noise = difusion_engine.add_noise(img_tensor, t)

        input_m3 = torch.cat([noisy_rgb, masked_tensor, boceto_predicho.detach(), mask_tensor], dim=1)

        optimizador_difusion.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred_noise = modelo_difusion(input_m3, t)
            
            loss_m3 = ((noise - pred_noise) ** 2 * mask_tensor).sum() / (mask_tensor.sum() + 1e-8)

        scaler.scale(loss_m3).backward()
        scaler.step(optimizador_difusion)
        scaler.update()

        # Acumular
        epoch_gen_loss    += gen_loss.item()
        epoch_dis_loss    += dis_loss.item()
        epoch_precision   += precision.item()
        loss_m3_acumulada += loss_m3.item()

        # logging
        if (batch_idx + 1) % config.LOG_INTERVAL == 0 or (batch_idx + 1) == len(cargaDatos):
            batches_hechos = batch_idx + 1
            elapsed        = time.time() - t_inicio_epoca
            seg_por_batch  = elapsed / batches_hechos
            restante       = seg_por_batch * (len(cargaDatos) - batches_hechos)
            print(
                f"  Batch {batches_hechos:>3}/{len(cargaDatos)} "
                f"| GenLoss: {epoch_gen_loss/batches_hechos:.4f} "
                f"| DisLoss: {epoch_dis_loss/batches_hechos:.4f} "
                f"| M3Loss: {loss_m3_acumulada/batches_hechos:.4f} "
                f"| ETA: {int(restante//60)}m {int(restante%60)}s"
            )

    num_batches  = len(cargaDatos)
    tiempo_epoca = time.time() - t_inicio_epoca
    print(f"\n  Resumen época {epoch}:")
    print(f"  Precisión Bordes M2: {(epoch_precision / num_batches) * 100:.2f}%")
    print(f"  Loss Difusión M3:    {loss_m3_acumulada / num_batches:.4f}")
    print(f"  Tiempo época:        {int(tiempo_epoca//60)}m {int(tiempo_epoca%60)}s")

    # FASE 3: EVALUACIÓN REVERSE DIFFUSION
    modelo_difusion.eval()
    
    with torch.no_grad():
        print("  Generando imagen final")
        imagen_reconstruida = difusion_engine.sample(modelo_difusion, masked_tensor, boceto_predicho, mask_tensor)

        try:
            psnr_val, ssim_val = calcular_metricas(img_tensor, imagen_reconstruida)
            print(f"  PSNR: {psnr_val:.4f} dB  |  SSIM: {ssim_val:.4f}")
            guardian.saveBest(modelo_difusion, ssim_val, epoch)
        except Exception as e:
            print(f"  [AVISO] Error en evaluación época {epoch}: {e}")

print("\n Entrenamiento Completo.")