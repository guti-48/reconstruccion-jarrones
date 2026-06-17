# Reconstruccion de ceramica antigua con GAN y difusion

Repositorio: https://github.com/guti-48/reconstruccion-jarrones.git

Este proyecto reconstruye zonas danadas en imagenes 2D de ceramica antigua. La ejecucion normal no requiere entrenar los modelos: los pesos entrenados ya estan incluidos en `checkpoints/`.

## Modulos del proyecto

- **M1 - datos y mascaras:** descarga las imagenes originales del conjunto Temasek Wreck, las redimensiona a 256x256 y genera mascaras sinteticas de dano. A partir de cada mascara crea tambien una imagen dañada que sirve como entrada del sistema.
- **M2 - generacion de bordes:** usa una GAN entrenada para completar la estructura del dibujo dentro de la zona dañada. El resultado es un esqueleto de bordes que guia la reconstruccion posterior.
- **M3 - difusion inversa:** usa una U-Net de difusion condicionada por la imagen dañada, la mascara y los bordes de M2. Su objetivo es reconstruir color y textura respetando la estructura propuesta por la GAN.

El flujo es escalable a otros conjuntos de imagenes: habria que preparar las imagenes con el mismo formato y reentrenar los modelos para que aprendan el nuevo dominio visual.

## Requisitos

Abrir una terminal en Visual Studio Code dentro de la carpeta del proyecto y ejecutar:

```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Ejecucion completa

### 1. Descargar las imagenes

```powershell
.\venv\Scripts\python.exe scripts_m1\descargaImagenes.py
```

Esto crea:

```text
data/raw/
```

### 2. Generar mascaras e imagenes danadas

```powershell
.\venv\Scripts\python.exe scripts_m1\preprocesamiento.py
```

Esto crea:

```text
data/processed/images/
data/processed/masks/
data/processed/masked_images/
```

### 3. Generar bordes con el modelo M2 entrenado

El repositorio ya incluye:

```text
checkpoints/EdgeModel_gen.pth
checkpoints/EdgeModel_dis.pth
```

Con esos pesos se generan los esqueletos de bordes:

```powershell
.\venv\Scripts\python.exe genera_bordes.py
```

Esto crea:

```text
data/results/edges/
```

### 4. Comprobar que todo esta preparado

```powershell
.\venv\Scripts\python.exe utils\validate_project.py --require-edges --require-checkpoints
```

La salida esperada debe indicar que existen imagenes, mascaras, imagenes danadas, bordes y checkpoints.

### 5. Ejecutar la reconstruccion visual

El repositorio ya incluye el checkpoint de difusion:

```text
checkpoints/modelo_m3_diffusion_epoch36_loss0.0247.pth
```

Para ver una reconstruccion:

```powershell
.\venv\Scripts\python.exe verImagen.py
```

Se abrira una ventana con cuatro imagenes: imagen danada, bordes generados, reconstruccion y original de referencia.

## Archivos principales

```text
scripts_m1/descargaImagenes.py      Descarga las imagenes originales
scripts_m1/preprocesamiento.py      Genera mascaras e imagenes danadas
genera_bordes.py                    Usa M2 entrenado para generar bordes
verImagen.py                        Ejecuta la reconstruccion final
utils/validate_project.py           Comprueba datos, bordes y checkpoints
checkpoints/                        Modelos ya entrenados
doc/documentacion_principal.pdf     Memoria del proyecto
```

## Nota sobre entrenamiento

Los scripts de entrenamiento se mantienen para reproducibilidad, pero no son necesarios para corregir o ejecutar la entrega:

```text
training/train_gan.py   Entrenamiento de M2
train_main.py           Entrenamiento de la difusion condicionada por M2
```
