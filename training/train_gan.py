import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.dataset import CeramicDataset
from utils.metrics import EdgeAccuracy
from utils.models import EdgeModel


class Config:
    PATH = "checkpoints"
    DATA_ROOT = "data/processed"
    GAN_LOSS = "nsgan"
    LR = 0.0001
    D2G_LR = 0.1
    BETA1 = 0.0
    BETA2 = 0.9
    FM_LOSS_WEIGHT = 10
    BATCH_SIZE = 4
    EPOCHS = 150
    EDGE_THRESHOLD = 0.5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    START_EPOCH = 1
    GEN_WEIGHTS = "EdgeModel_gen.pth"
    DIS_WEIGHTS = "EdgeModel_dis.pth"


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento M2: GAN generadora de bordes.")
    parser.add_argument("--data-root", default=Config.DATA_ROOT, help="Carpeta data/processed.")
    parser.add_argument("--checkpoints-dir", default=Config.PATH, help="Carpeta donde guardar pesos.")
    parser.add_argument("--epochs", type=int, default=Config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=Config.LR)
    parser.add_argument("--start-epoch", type=int, default=Config.START_EPOCH)
    parser.add_argument("--resume", action="store_true", help="Retoma EdgeModel_gen.pth y EdgeModel_dis.pth si existen.")
    return parser.parse_args()


def load_previous_weights(edge_model, config):
    ruta_gen = os.path.join(config.PATH, config.GEN_WEIGHTS)
    ruta_dis = os.path.join(config.PATH, config.DIS_WEIGHTS)

    if not os.path.exists(ruta_gen) or not os.path.exists(ruta_dis):
        print("No se encontraron pesos previos de M2. Se empieza desde cero.")
        config.START_EPOCH = 1
        return

    checkpoint_gen = torch.load(ruta_gen, map_location=config.DEVICE)
    checkpoint_dis = torch.load(ruta_dis, map_location=config.DEVICE)

    pesos_gen = checkpoint_gen["generator"] if "generator" in checkpoint_gen else checkpoint_gen
    pesos_dis = checkpoint_dis["discriminator"] if "discriminator" in checkpoint_dis else checkpoint_dis

    edge_model.generator.load_state_dict(pesos_gen)
    edge_model.discriminator.load_state_dict(pesos_dis)
    print(f"Pesos previos de M2 cargados desde {config.PATH}.")


def train_gan():
    args = parse_args()
    config = Config()
    config.DATA_ROOT = args.data_root
    config.PATH = args.checkpoints_dir
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LR = args.lr
    config.START_EPOCH = args.start_epoch if args.resume else 1

    os.makedirs(config.PATH, exist_ok=True)

    print("--- Entrenamiento M2: Edge GAN ---")
    print(f"Dispositivo: {config.DEVICE}")
    print(f"Dataset: {config.DATA_ROOT}")
    print(f"Checkpoints: {config.PATH}")

    train_dataset = CeramicDataset(root_dir=config.DATA_ROOT, edge_mode="canny")
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    if len(train_loader) == 0:
        raise RuntimeError("El DataLoader no tiene batches. Baja --batch-size o revisa el dataset.")

    edge_model = EdgeModel(config).to(config.DEVICE)
    edgeacc = EdgeAccuracy(config.EDGE_THRESHOLD).to(config.DEVICE)

    if args.resume:
        print(f"Retomando entrenamiento desde la epoca {config.START_EPOCH}.")
        load_previous_weights(edge_model, config)
    else:
        print("Entrenamiento desde cero.")

    for epoch in range(config.START_EPOCH, config.EPOCHS + 1):
        print(f"\nEpoca {epoch}/{config.EPOCHS}")
        edge_model.train()

        epoch_gen_loss = 0.0
        epoch_dis_loss = 0.0
        epoch_precision = 0.0

        for items in train_loader:
            _, _, img_gray, edges, masks = [item.to(config.DEVICE) for item in items]

            outputs, gen_loss, dis_loss, _ = edge_model.process(img_gray, edges, masks)
            precision, _ = edgeacc(edges * masks, outputs * masks)
            edge_model.backward(gen_loss, dis_loss)

            epoch_gen_loss += gen_loss.item()
            epoch_dis_loss += dis_loss.item()
            epoch_precision += precision.item()

        num_batches = len(train_loader)
        print(f" -> Perdida generador: {epoch_gen_loss / num_batches:.4f}")
        print(f" -> Perdida discriminador: {epoch_dis_loss / num_batches:.4f}")
        print(f" -> Precision de bordes: {(epoch_precision / num_batches) * 100:.2f}%")

        if epoch % 10 == 0:
            edge_model.save()
            print(f"Checkpoint guardado en la epoca {epoch}.")

    edge_model.save()
    print("\nEntrenamiento GAN completado.")


if __name__ == "__main__":
    train_gan()
