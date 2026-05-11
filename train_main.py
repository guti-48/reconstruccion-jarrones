import torch, os, time, glob, re
import torch.nn as nn
from torch.utils.data import DataLoader

# Importaciones M1 (Datos y Orquestación)
from utils.dataset import CeramicDataset
from utils.metrics import calcular_metricas
from utils.checkpoint import CheckpointManager

# Importaciones M3 (Difusión de Oussama)
from models.unet import UNet
from models.diffusion import Diffusion 

# CONFIGURACIÓN Y RUTAS (PREPARADO PARA COLAB)
dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Hardware detectado: {dispositivo}")

class Config:
    DEVICE         = dispositivo
    PATH           = '/content/drive/MyDrive/Articulo-Investigacion/checkpoints'
    #PATH           = 'checkpoints'
    LR             = 1e-5
    BATCH_SIZE     = 16  
    EPOCHS         = 150
    NUM_WORKERS    = 2  
    LOG_INTERVAL   = 10
    DIFFUSION_T    = 200 
    
    # Rutas adaptadas a tu estructura (ajustables para Colab)
    # ANTICIPACIÓN DE FALLO: Centralizamos las rutas aquí para no rebuscar en el código si cambian en Drive
    ROOT_DIR       = '/content/data_rapida/processed' 
    EDGES_DIR      = '/content/data_rapida/results/edges' 

    #ROOT_DIR       = 'data/processed' 
    #EDGES_DIR      = 'data/results/edges'

config = Config()

# 1. CARGA DE DATOS (EL PUENTE M2 -> M3)
# Inyectamos explícitamente la ruta de los bordes GAN al Dataset
dataset = CeramicDataset(root_dir=config.ROOT_DIR, edges_dir=config.EDGES_DIR)

cargaDatos = DataLoader(
    dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=config.NUM_WORKERS,
    pin_memory=(dispositivo.type == 'cuda')
)
print(f"[INFO] Dataset: {len(dataset)} imágenes | {len(cargaDatos)} batches por época")

guardian = CheckpointManager(save_dir=config.PATH, model_name='modelo_m3_diffusion')

# ==========================================
# 2. INICIALIZACIÓN M3 (OUSSAMA)
# ==========================================
# UNet condicionada a 8 canales: 3(Ruido) + 3(Dañada) + 1(Bordes GAN) + 1(Máscara)
modelo_difusion = UNet(in_channels=8, out_channels=3).to(dispositivo)
difusion_engine = Diffusion(T=config.DIFFUSION_T, device=dispositivo)

optimizador_difusion = torch.optim.Adam(modelo_difusion.parameters(), lr=config.LR)

# 3. RECUPERACIÓN DE CHECKPOINTS
patron_busqueda = os.path.join(config.PATH, 'modelo_m3_diffusion_epoch*.pth')
archivos_guardados = glob.glob(patron_busqueda)

if archivos_guardados:
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

    match = re.search(r'ssim([0-9]+\.[0-9]+)', nombre_archivo)
    match = re.search(r'loss([0-9]+\.[0-9]+)', nombre_archivo)
    if match:
        guardian.best_metric = float(match.group(1))
        print(f"  -> [INFO] Guardián actualizado. Récord histórico a batir: {guardian.best_metric:.4f}")
else:
    print("  -> [AVISO] No hay checkpoints previos. Entrenando Difusión (M3) desde cero.")

use_amp = (dispositivo.type == 'cuda')
scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

# 4. BUCLE DE ENTRENAMIENTO PRINCIPAL
for epoch in range(1, config.EPOCHS + 1):
    print(f"\n{'='*50}\n  Época {epoch}/{config.EPOCHS}\n{'='*50}")
    modelo_difusion.train()

    loss_m3_acumulada = 0.0
    t_inicio_epoca    = time.time()

    for batch_idx, items in enumerate(cargaDatos):

        masked_tensor, img_tensor, _, edges_tensor, mask_tensor = [
            item.to(dispositivo) for item in items
        ]

        t = torch.randint(0, difusion_engine.T, (img_tensor.shape[0],), device=dispositivo).long()

        noisy_rgb, noise = difusion_engine.add_noise(img_tensor, t)

        input_m3 = torch.cat([noisy_rgb, masked_tensor, edges_tensor, mask_tensor], dim=1)

        optimizador_difusion.zero_grad()

        with torch.amp.autocast('cuda', enabled=use_amp):
            pred_noise = modelo_difusion(input_m3, t)
            
            mse_loss = (noise - pred_noise) ** 2

            mapa_pesos = 1.0 + (9.0 * mask_tensor)

            mapa_pesos[:, :, 180:, :] = 0.0 

            loss_m3 = (mse_loss * mapa_pesos).sum() / (mapa_pesos.sum() + 1e-8)

        scaler.scale(loss_m3).backward()
        scaler.step(optimizador_difusion)
        scaler.update()

        loss_m3_acumulada += loss_m3.item()

        if (batch_idx + 1) % config.LOG_INTERVAL == 0 or (batch_idx + 1) == len(cargaDatos):
            batches_hechos = batch_idx + 1
            elapsed        = time.time() - t_inicio_epoca
            seg_por_batch  = elapsed / batches_hechos
            restante       = seg_por_batch * (len(cargaDatos) - batches_hechos)
            print(
                f"  Batch {batches_hechos:>3}/{len(cargaDatos)} "
                f"| M3Loss (MSE): {loss_m3_acumulada/batches_hechos:.4f} "
                f"| ETA: {int(restante//60)}m {int(restante%60)}s"
            )

    tiempo_epoca = time.time() - t_inicio_epoca
    print(f"\n  Resumen época {epoch} | Loss: {loss_m3_acumulada / len(cargaDatos):.4f} | Tiempo: {int(tiempo_epoca//60)}m {int(tiempo_epoca%60)}s")

 
    # 5. FASE DE EVALUACIÓN 
    modelo_difusion.eval()
    
    with torch.no_grad():
        print("  Generando muestra para validación...")
        img_eval     = img_tensor[0:1]
        masked_eval  = masked_tensor[0:1]
        edges_eval   = edges_tensor[0:1]
        mask_eval    = mask_tensor[0:1]

        imagen_reconstruida = difusion_engine.sample(modelo_difusion, masked_eval, edges_eval, mask_eval)

        try:
            mask_rgb = mask_eval.expand_as(img_eval).clone() # Clonamos para no alterar el tensor original

            mask_rgb[:, :, 180:, :] = 0

            pixeles_falsos = imagen_reconstruida[mask_rgb == 1]
            pixeles_reales = img_eval[mask_rgb == 1]

            if pixeles_falsos.numel() > 0:
                error_agujero = torch.nn.functional.l1_loss(pixeles_falsos, pixeles_reales).item()
                print(f"  Error L1 (Estricto en Cerámica): {error_agujero:.4f}")
                guardian.saveBest(modelo_difusion, error_agujero, epoch)
            else:
                print("  [AVISO] La máscara estaba completamente en la zona ignorada. Se salta validación.")

        except Exception as e:
            print(f"  [AVISO CRÍTICO] Error en cálculo de métricas: {e}")

print("\n[INFO] Entrenamiento Completo y a salvo.")