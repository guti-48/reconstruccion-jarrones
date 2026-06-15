import argparse
import glob
import os
import re
import time

import torch
from torch.utils.data import DataLoader

from models.diffusion import Diffusion
from models.unet import UNet
from utils.checkpoint import CheckpointManager
from utils.dataset import CeramicDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenamiento global M2(Rafa/bordes) + M3(Oussama/difusion)."
    )
    parser.add_argument("--root-dir", default="data/processed")
    parser.add_argument("--edges-dir", default="data/results/edges")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--edge-mode", choices=["generated", "canny"], default="generated")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--diffusion-t", type=int, default=200)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser.parse_args()


def latest_diffusion_checkpoint(checkpoints_dir):
    pattern = os.path.join(checkpoints_dir, "modelo_m3_diffusion_epoch*.pth")
    candidates = glob.glob(pattern)
    if not candidates:
        fallback = os.path.join(checkpoints_dir, "diffusion_best.pth")
        return fallback if os.path.exists(fallback) else None
    return max(candidates, key=os.path.getmtime)


def load_model_state(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)


def infer_best_loss(checkpoint_path):
    match = re.search(r"loss([0-9]+(?:\.[0-9]+)?)", os.path.basename(checkpoint_path))
    return float(match.group(1)) if match else None


def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print(f"[INFO] Hardware detectado: {device}")
    print(f"[INFO] Dataset: {args.root_dir}")
    print(f"[INFO] Bordes M2: {args.edge_mode} ({args.edges_dir})")

    dataset = CeramicDataset(
        root_dir=args.root_dir,
        edges_dir=args.edges_dir,
        edge_mode=args.edge_mode,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=use_amp,
    )

    if len(dataloader) == 0:
        raise RuntimeError("El DataLoader no tiene batches. Baja --batch-size o revisa el dataset.")

    print(f"[INFO] Dataset: {len(dataset)} imagenes | {len(dataloader)} batches por epoca")

    model = UNet(in_channels=8, out_channels=3).to(device)
    diffusion = Diffusion(T=args.diffusion_t, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    guardian = CheckpointManager(save_dir=args.checkpoints_dir, model_name="modelo_m3_diffusion")

    if args.resume:
        checkpoint_path = latest_diffusion_checkpoint(args.checkpoints_dir)
        if checkpoint_path:
            load_model_state(model, checkpoint_path, device)
            best_loss = infer_best_loss(checkpoint_path)
            if best_loss is not None:
                guardian.best_metric = best_loss
            print(f"[INFO] M3 retomado desde: {os.path.basename(checkpoint_path)}")
        else:
            print("[INFO] No hay checkpoint M3 previo. Entrenando desde cero.")

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'=' * 50}\n  Epoca {epoch}/{args.epochs}\n{'=' * 50}")
        model.train()

        epoch_loss = 0.0
        start = time.time()

        for batch_idx, items in enumerate(dataloader):
            masked, real, _, edges, mask = [item.to(device) for item in items]
            t = torch.randint(0, diffusion.T, (real.shape[0],), device=device).long()
            noisy_rgb, noise = diffusion.add_noise(real, t)
            model_input = torch.cat([noisy_rgb, masked, edges, mask], dim=1)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred_noise = model(model_input, t)
                mse = (noise - pred_noise) ** 2
                weights = 1.0 + (9.0 * mask)
                weights[:, :, 180:, :] = 0.0
                loss = (mse * weights).sum() / (weights.expand_as(mse).sum() + 1e-8)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            batches_done = batch_idx + 1
            if batches_done % args.log_interval == 0 or batches_done == len(dataloader):
                elapsed = time.time() - start
                seconds_per_batch = elapsed / batches_done
                remaining = seconds_per_batch * (len(dataloader) - batches_done)
                print(
                    f"  Batch {batches_done:>3}/{len(dataloader)} "
                    f"| M3Loss: {epoch_loss / batches_done:.4f} "
                    f"| ETA: {int(remaining // 60)}m {int(remaining % 60)}s"
                )

        avg_loss = epoch_loss / len(dataloader)
        elapsed = time.time() - start
        print(f"\n  Resumen epoca {epoch} | Loss: {avg_loss:.4f} | Tiempo: {int(elapsed // 60)}m {int(elapsed % 60)}s")

        model.eval()
        with torch.no_grad():
            img_eval = real[0:1]
            masked_eval = masked[0:1]
            edges_eval = edges[0:1]
            mask_eval = mask[0:1]
            reconstructed = diffusion.sample(model, masked_eval, edges_eval, mask_eval)

            mask_rgb = mask_eval.expand_as(img_eval).clone()
            mask_rgb[:, :, 180:, :] = 0
            fake_pixels = reconstructed[mask_rgb == 1]
            real_pixels = img_eval[mask_rgb == 1]

            if fake_pixels.numel() > 0:
                hole_l1 = torch.nn.functional.l1_loss(fake_pixels, real_pixels).item()
                print(f"  Error L1 en agujero: {hole_l1:.4f}")
                guardian.saveBest(model, hole_l1, epoch)
            else:
                print("  [AVISO] La mascara cae en la zona ignorada. Se salta validacion.")

    print("\n[INFO] Entrenamiento global completado.")


if __name__ == "__main__":
    train()