# Prediccion de areas faltantes en ceramica antigua

Repositorio: https://github.com/guti-48/reconstruccion-jarrones.git

Proyecto de Procesamiento de Imagenes Digitales orientado a reconstruir zonas perdidas en fotografias 2D de ceramica antigua. El flujo final combina:

- **M1 - datos y mascaras:** descarga imagenes del Temasek Wreck, las redimensiona a 256x256 y genera danos sinteticos.
- **M2 - bordes:** entrena una GAN tipo EdgeConnect para completar la estructura o esqueleto de la pieza.
- **Entrenamiento conjunto M2 -> M3:** parte del M2 ya entrenado, genera los bordes intermedios y entrena en Colab la difusion inversa que reconstruye color y textura.

La memoria ampliada de la entrega esta en [doc/memoria_mejorada.md](doc/memoria_mejorada.md).

## Estructura

```text
Articulo-Investigacion/
|-- data/
|   |-- raw/                  # Imagenes descargadas originales
|   |-- processed/
|   |   |-- images/           # Imagen limpia 256x256
|   |   |-- masks/            # Mascara binaria: 1 = zona danada
|   |   `-- masked_images/    # Imagen con dano sintetico
|   `-- results/
|       `-- edges/            # Bordes generados por M2
|-- checkpoints/              # Pesos entrenados
|-- doc/
|   |-- memoria_mejorada.md   # Memoria revisada
|   `-- figures/              # Figuras usadas en la memoria
|-- models/                   # U-Net, difusion y redes base
|-- scripts_m1/               # Descarga y preprocesamiento
|-- training/                 # Entrenamiento de M2
|-- utils/                    # Dataset, metricas, checkpoints y validacion
|-- test/                     # Pruebas rapidas
|-- genera_bordes.py          # Genera esqueletos con M2 entrenado
|-- train_main.py             # Entrenamiento conjunto M2 -> M3
`-- verImagen.py              # Inferencia visual final
```

## Instalacion

Se recomienda Python 3.10 u 3.11. En Colab se puede usar el runtime GPU y montar Drive para guardar checkpoints.

Windows:

```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
>>>>>>> f734264 (organizcion del respsitorio)
```

Linux, macOS o Colab:

```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Las dependencias estan fijadas en `requirements.txt` para que el entorno sea reproducible.

## Flujo final recomendado

### 1. Preparar datos

Si `data/processed` ya existe y contiene datos, este paso puede saltarse.

```powershell
python scripts_m1\descargaImagenes.py
python scripts_m1\preprocesamiento.py
```

Salida esperada:

```text
data/processed/images/
data/processed/masks/
data/processed/masked_images/
```

### 2. Validar dataset base

```powershell
python utils\validate_project.py
```

En el estado actual del proyecto local hay 640 imagenes, 640 mascaras y 640 imagenes danadas.

### 3. Entrenar M2, modelo de bordes

M2 se entrena con `edge_mode="canny"` dentro del dataset. Esto significa que el objetivo de entrenamiento son bordes Canny calculados desde la imagen limpia, no bordes generados por la propia GAN.

```powershell
python training\train_gan.py --epochs 150 --batch-size 4
```

Salida esperada:

```text
checkpoints/EdgeModel_gen.pth
checkpoints/EdgeModel_dis.pth
```

Para retomar un entrenamiento existente:

```powershell
python training\train_gan.py --resume --start-epoch 51 --epochs 150 --batch-size 4
```

Para Colab se pueden pasar rutas explicitas:

```bash
python training/train_gan.py \
  --data-root /content/data_rapida/processed \
  --checkpoints-dir /content/drive/MyDrive/Articulo-Investigacion/checkpoints \
  --epochs 150 \
  --batch-size 16
```

### 4. Generar bordes M2

Cuando exista `checkpoints/EdgeModel_gen.pth`, se generan los esqueletos que usara la difusion.

```powershell
python genera_bordes.py
```

Salida esperada:

```text
data/results/edges/esqueleto_<nombre_imagen>.jpg
```

### 5. Validar intermedios y checkpoints

Antes de entrenar la difusion integrada en modo final:

```powershell
python utils\validate_project.py --require-edges
```

Antes de hacer inferencia con pesos ya entrenados:

```powershell
python utils\validate_project.py --require-edges --require-checkpoints
```

### 6. Entrenar la difusion integrada M2 -> M3

Este es el entrenamiento final recomendado. No parte de un M3 ya entrenado: parte del M2 entrenado, usa sus bordes generados como condicion y entrena la U-Net de difusion en Colab o local. La entrada combina imagen ruidosa RGB, imagen danada RGB, borde generado por M2 y mascara.

```powershell
python train_main.py --epochs 150 --batch-size 4 --num-workers 0
```

Si hay GPU:

```powershell
python train_main.py --epochs 150 --batch-size 16 --num-workers 2
```

El script retoma automaticamente el mejor checkpoint de difusion si existe:

```text
checkpoints/modelo_m3_diffusion_epoch*_loss*.pth
```

Para entrenar desde cero:

```powershell
python train_main.py --no-resume --epochs 150 --batch-size 4 --num-workers 0
```

Ejemplo en Colab:

```bash
python train_main.py \
  --root-dir /content/data_rapida/processed \
  --edges-dir /content/data_rapida/results/edges \
  --checkpoints-dir /content/drive/MyDrive/Articulo-Investigacion/checkpoints \
  --epochs 150 \
  --batch-size 16 \
  --num-workers 2
```

### 7. Inferencia visual

```powershell
python verImagen.py
```

La salida compara imagen danada, bordes, reconstruccion y original.

## Scripts finales

| Script | Estado | Uso |
| --- | --- | --- |
| `scripts_m1/descargaImagenes.py` | Final | Descarga imagenes originales. |
| `scripts_m1/preprocesamiento.py` | Final | Genera `data/processed`. |
| `training/train_gan.py` | Final | Entrena M2. |
| `genera_bordes.py` | Final | Genera `data/results/edges`. |
| `utils/validate_project.py` | Final | Comprueba datos, bordes y checkpoints. |
| `train_main.py` | Final | Entrena la difusion partiendo de bordes generados por M2. |
| `verImagen.py` | Final | Ejecuta inferencia visual. |

## Contrato del dataset

`utils/dataset.py` define `CeramicDataset`. Devuelve cinco tensores:

```text
masked_rgb, original_rgb, original_gray, edges, mask
```

Modos de bordes:

- `edge_mode="canny"`: genera bordes Canny al vuelo. Es el modo correcto para entrenar M2.
- `edge_mode="generated"`: carga `data/results/edges/esqueleto_*`. Es el modo correcto para entrenar la difusion integrada.

## Comandos de comprobacion

Compilar scripts principales:

```powershell
python -m py_compile utils\dataset.py utils\validate_project.py training\train_gan.py train_main.py genera_bordes.py
```

Probar arranque del entrenamiento integrado sin entrenar:

```powershell
python train_main.py --epochs 0 --batch-size 2 --num-workers 0
```

Ejecutar tests rapidos:

```powershell
python test\test_networks.py
python test\test_dataset.py
python test\test_canny.py
```

## Mejoras de la revision

- README reorganizado con URL del repositorio y flujo final sin scripts alternativos ambiguos.
- Memoria ampliada en `doc/memoria_mejorada.md` con figuras, referencias, experimentacion y reparto de aportaciones.
- `requirements.txt` con versiones fijadas.
- `training/train_gan.py` alineado con rutas locales por defecto y argumentos para Colab.
- `utils/validate_project.py` para garantizar que datos, bordes y checkpoints existen antes de entrenar o inferir.
