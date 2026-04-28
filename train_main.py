import torch, os, time
import torch.nn as nn
from torch.utils.data import DataLoader

# Importaciones M1 (guti)
from utils.dataset import CeramicDataset
from utils.metrics import calcular_metricas, EdgeAccuracy
from utils.checkpoint import CheckpointManager
from models.diffusion import add_noise

# Importaciones M2 (rafa)
from utils.models import EdgeModel

# Importaciones M3 (oussama)
from models.unet import UNet

# ============================================================
# CONFIGURACIÓN GLOBAL
# ============================================================
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

config = Config()

# ============================================================
# DATOS
# ============================================================
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

# ============================================================
# CHECKPOINT MANAGER
# ============================================================
guardian = CheckpointManager(save_dir='checkpoints', model_name='modeloConjunto')

# ============================================================
# MODELOS
# ============================================================
print("Cargando EdgeModel (M2) y UNet de Difusión (M3)...")
modelo_bordes   = EdgeModel(config).to(dispositivo)
modelo_difusion = UNet(in_channels=4, out_channels=3).to(dispositivo)

optimizador_difusion = torch.optim.Adam(modelo_difusion.parameters(), lr=config.LR)
criterio_difusion    = nn.MSELoss()

edgeacc = EdgeAccuracy(threshold=config.EDGE_THRESHOLD)

#si hay GPU, usar autocast para precisión mixta
use_amp = (dispositivo.type == 'cuda')
scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

# ============================================================
# BUCLE DE ENTRENAMIENTO
# ============================================================
for epoch in range(1, config.EPOCHS + 1):
    print(f"\n{'='*50}")
    print(f"  Época {epoch}/{config.EPOCHS}")
    print(f"{'='*50}")

    modelo_bordes.train()
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

        # FASE 1: GAN DE BORDES (Rafa)
        with torch.cuda.amp.autocast(enabled=use_amp):
            boceto_predicho, gen_loss, dis_loss, logs_m2 = modelo_bordes.process(
                img_gray_tensor, edges_tensor, mask_tensor
            )

        precision, recall = edgeacc(
            edges_tensor * mask_tensor,
            boceto_predicho * mask_tensor
        )
        modelo_bordes.backward(gen_loss, dis_loss)

        # FASE 2: DIFUSIÓN (OUssama)
        t = torch.randint(1, 1000, (img_tensor.shape[0], 1, 1, 1)).to(dispositivo)
        noisy_rgb, noise = add_noise(img_tensor, t)

        input_m3 = torch.cat([noisy_rgb, boceto_predicho.detach()], dim=1)

        optimizador_difusion.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred_noise = modelo_difusion(input_m3)
            loss_m3    = criterio_difusion(pred_noise, noise)

        scaler.scale(loss_m3).backward()
        scaler.step(optimizador_difusion)
        scaler.update()

        #acumular
        epoch_gen_loss    += gen_loss.item()
        epoch_dis_loss    += dis_loss.item()
        epoch_precision   += precision.item()
        loss_m3_acumulada += loss_m3.item()

        # asi veo que no se queda pillau 
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
    print(f"  Loss GAN Gen:        {epoch_gen_loss  / num_batches:.4f}")
    print(f"  Loss GAN Dis:        {epoch_dis_loss  / num_batches:.4f}")
    print(f"  Precisión Bordes M2: {(epoch_precision / num_batches) * 100:.2f}%")
    print(f"  Loss Difusión M3:    {loss_m3_acumulada / num_batches:.4f}")
    print(f"  Tiempo época:        {int(tiempo_epoca//60)}m {int(tiempo_epoca%60)}s")

    # FASE 3: evaluacion y guardado
    modelo_difusion.eval()
    modelo_bordes.eval()

    with torch.no_grad():
        # oussama tiene que terminar su parte, esto es solo orientativo hasta que su parte accabe
        #     imagen_reconstruida = reverse_diffusion(modelo_difusion, masked_tensor, boceto_predicho, dispositivo)
        imagen_reconstruida = masked_tensor
        # ─────────────────────────────────────────────────────────────────────

        try:
            psnr_val, ssim_val = calcular_metricas(img_tensor, imagen_reconstruida)
            print(f"  PSNR: {psnr_val:.4f} dB  |  SSIM: {ssim_val:.4f}")
            guardian.saveBest(modelo_difusion, ssim_val, epoch)
            modelo_bordes.save()
        except Exception as e:
            print(f"  [AVISO] Error en evaluación época {epoch}: {e}")

print("\n✅ Entrenamiento Completo.")