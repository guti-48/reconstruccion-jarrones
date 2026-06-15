import argparse
import glob
import os
import sys


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def count_images(path):
    if not os.path.isdir(path):
        return None
    return len([name for name in os.listdir(path) if name.lower().endswith(IMAGE_EXTENSIONS)])


def require_dir(path):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"No existe el directorio requerido: {path}")


def require_file(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No existe el archivo requerido: {path}")


def validate(args):
    images_dir = os.path.join(args.root_dir, "images")
    masks_dir = os.path.join(args.root_dir, "masks")
    masked_dir = os.path.join(args.root_dir, "masked_images")

    for path in (images_dir, masks_dir, masked_dir):
        require_dir(path)

    image_count = count_images(images_dir)
    mask_count = count_images(masks_dir)
    masked_count = count_images(masked_dir)

    print(f"Imagenes limpias: {image_count}")
    print(f"Mascaras: {mask_count}")
    print(f"Imagenes danadas: {masked_count}")

    if len({image_count, mask_count, masked_count}) != 1:
        raise RuntimeError("Los conteos de images/masks/masked_images no coinciden.")

    if args.require_edges:
        require_dir(args.edges_dir)
        edge_count = count_images(args.edges_dir)
        print(f"Bordes generados M2: {edge_count}")
        if edge_count != image_count:
            raise RuntimeError("El numero de bordes generados no coincide con el dataset procesado.")

        missing_edges = []
        for image_path in glob.glob(os.path.join(images_dir, "*")):
            name = os.path.basename(image_path)
            base_name = os.path.splitext(name)[0]
            pattern = os.path.join(args.edges_dir, f"esqueleto_{base_name}.*")
            if not glob.glob(pattern):
                missing_edges.append(name)

        if missing_edges:
            preview = ", ".join(missing_edges[:5])
            raise RuntimeError(f"Faltan bordes para {len(missing_edges)} imagenes. Ejemplos: {preview}")

    if args.require_checkpoints:
        require_file(os.path.join(args.checkpoints_dir, "EdgeModel_gen.pth"))
        require_file(os.path.join(args.checkpoints_dir, "EdgeModel_dis.pth"))
        diffusion_pattern = os.path.join(args.checkpoints_dir, "modelo_m3_diffusion_epoch*_loss*.pth")
        if not glob.glob(diffusion_pattern):
            raise FileNotFoundError(f"No se encontro checkpoint M3 con patron: {diffusion_pattern}")
        print("Checkpoints de M2 y difusion encontrados.")

    print("Validacion completada correctamente.")


def parse_args():
    parser = argparse.ArgumentParser(description="Comprueba datos, bordes intermedios y checkpoints del proyecto.")
    parser.add_argument("--root-dir", default="data/processed")
    parser.add_argument("--edges-dir", default="data/results/edges")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--require-edges", action="store_true", help="Exige bordes generados por M2.")
    parser.add_argument("--require-checkpoints", action="store_true", help="Exige pesos entrenados de M2 y M3.")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        validate(parse_args())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
