# Predicción Faltante de Áreas en vasijas de Cerámica

Este repositorio contiene la implementació del Trabajo Dirigido de Procesamiento de Imagenes Digitales. El objetivo del proyecto es reconstruir digitalmente las áreas faltantes, ya sea roturas o desgastes, en imágenes 2D de procelana antigua utilzando un enfoque dual: Predicción de Patrones (Deep Adversial) y Predicción de color/textura (Reverser Diffusion).

Este proyecto esta basado en el paper "Digital prediction of ancient ceramic images missing areas based on deep adversarial and reverse diffusion" 

## Estructura del proyecto

Nos moveremos a lo laro de varias entregas, tendremos que añadir la primera

El desarrollo de esta segunda estapa está dividido en tres módulos principales, correspondientes a las responsabilidades del equipo:

- **Módulo 1: Datos y Procesamiento** Extracción, filtrado y generacion de datos sintéticos
- **Módulo 2: Estrcutura**  Generación de bordes mediante `edge-connection`
- **Módulo 3: Textura**  Relleno de color mediante la difusión inversa con `rePaint`



## Ejecución código para esta fase

Crearemos un entorno virtual venv

```bash
    python -m venv venv
    source ./venv/Scripts/activate
    python descargaImagenes.py
    python preprocesamiento.py
```

Con los comandos anteriores obtendremos la selección de imagenes que usaremos como referencias, y a esa selección aplicada una mascára binaria en la que aplicaremos nuestros algoritmos

---

## 📋 Módulos Python y su función

### **1. `descargaImagenes.py` — Recopilación de datos**
- **Propósito:** Descargar dataset de cerámica antigua desde servidor remoto
- **Entrada:** CSV con metadatos (`temasekwreck-temasekblueandwhites.csv`)
- **Proceso:**
  - Filtra piezas por estado (Intact, Half intact, Base, Rim to Base)
  - Filtra por forma sencilla (Dish, Bowl & dish)
  - Descarga imágenes .jpg desde `https://epress.nus.edu.sg/sitereports/temasekwreck/`
  - Maneja reintentos automáticos y timeouts
- **Salida:** Carpeta `img/` con imágenes descargadas
- **Ejecución:**
  ```bash
  python descargaImagenes.py
  ```

### **2. `preprocesamiento.py` — Preparación de dataset**
- **Propósito:** Redimensionar imágenes y generar máscaras de daño sintético
- **Entrada:** Imágenes de `img/`
- **Proceso:**
  1. Lee cada imagen .jpg
  2. Redimensiona a 256x256 (estándar para redes neuronales)
  3. Genera máscara binaria con daño artificial:
     - Dibuja grietas (líneas aleatorias)
     - Añade zonas dañadas (elipses)
  4. Crea imagen con daño visible (píxeles blancos en zonas dañadas)
  5. Guarda 3 versiones:
     - `images/` → original intacta
     - `masks/` → máscara del daño (255=daño, 0=intacto)
     - `masked_images/` → original CON daño aplicado
- **Salida:** Carpeta `imagenes_procesadas/` con 3 subdirectorios
- **Ejecución:**
  ```bash
  python preprocesamiento.py
  ```

### **3. `utils/dataset.py` — Cargador de datos (Dataset PyTorch)**
- **Propósito:** Interfaz para acceder a imágenes durante entrenamiento
- **Clase:** `CeramicDataset`
- **Funcionalidad:**
  - Lee pares (imagen_dañada, imagen_original)
  - Normaliza a valores [0, 1]
  - Convierte a tensores PyTorch (B, 3, 256, 256)
  - Compatible con `DataLoader` para batches
- **No se ejecuta directamente**, se importa en `train_diffusion.py`

### **4. `models/unet.py` — Arquitectura de red neuronal**
- **Propósito:** Definir el modelo U-Net para predicción de ruido
- **Clase:** `UNet(in_channels=3, out_channels=3)`
- **Arquitectura:**
  ```
  Input (B, 3, 256, 256)
    ↓
  Encoder 1: Conv2d(3→64) + ReLU
    ↓ MaxPool(2x2)
  Encoder 2: Conv2d(64→128) + ReLU
    ↓ Upsample(2x2)
  Decoder 1: Conv2d(192→64) + ReLU [concatena skip connection]
    ↓
  Output: Conv2d(64→3)
    ↓
  Output (B, 3, 256, 256)
  ```
- **Función:** Predice ruido gaussiano (para entrenar modelo difusión)
- **No se ejecuta directamente**, se carga en `train_diffusion.py`

### **5. `models/diffusion.py` — Funciones de difusión**
- **Propósito:** Implementar forward diffusion (corrupción con ruido)
- **Función principal:** `add_noise(x, t)`
  - Implementa ecuación DDPM: $x_t = \sqrt{\alpha_t} \cdot x_0 + \sqrt{1-\alpha_t} \cdot \epsilon$
  - `α_t = 0.9^t` (decae exponencialmente)
  - `ε ~ N(0,1)` (ruido gaussiano)
  - **Entrada:** imagen original + timestep t
  - **Salida:** imagen ruidosa + ruido gaussiano
- **Uso:** Generar datos de entrenamiento corrompidos
- **No se ejecuta directamente**, se importa en `train_diffusion.py`

### **6. `utils/visualize.py` — Visualización de resultados**
- **Propósito:** Mostrar comparación visual de reconstrucciones
- **Función:** `mostrar_resultados(masked, real, model)`
- **Proceso:**
  1. Carga modelo en modo evaluación
  2. Aplica ruido a imagen original
  3. Predice ruido con el modelo
  4. Reconstruye: `recon = noisy - pred_noise`
  5. Muestra 3 gráficas lado a lado:
     - Imagen dañada
     - Original intacta
     - Reconstrucción estimada
- **No se ejecuta directamente**, se llama desde `train_diffusion.py` cada epoch

### **7. `training/train_diffusion.py` — Script de entrenamiento**
- **Propósito:** Entrenar el modelo U-Net para predecir ruido (reverse diffusion)
- **Flujo de ejecución:**
  ```
  1. Crear Dataset y DataLoader desde imagenes_procesadas/
  2. Instanciar U-Net y optimizer
  3. Loop por 5 epochs:
     a) Para cada batch (masked, real):
        - Sample timestep aleatorio: t ∈ [1..10]
        - Añadir ruido: noisy, noise_gt = add_noise(real, t)
        - Predecir: pred_noise = model(noisy)
        - Calcular loss: MSE(pred_noise, noise_gt)
        - Backprop y actualizar pesos
     b) Imprimir loss promedio del epoch
     c) Visualizar reconstrucción con mostrar_resultados()
  4. Guardar modelo entrenado (opcional)
  ```
- **Parámetros:**
  - `batch_size=4`
  - `learning_rate=1e-4`
  - `epochs=5`
  - `timesteps=[1..10]`
- **Ejecución:**
  ```bash
  cd training
  python train_diffusion.py
  ```

---

## 🔄 FLUJO COMPLETO DEL PROYECTO

```
┌─────────────────────────────────────────────────────────────┐
│ FASE 1: RECOPILACIÓN Y PREPARACIÓN DE DATOS               │
└─────────────────────────────────────────────────────────────┘
        │
        ├─→ descargaImagenes.py
        │   └─ Lee CSV → Filtra → Descarga imágenes → /img/
        │
        ├─→ preprocesamiento.py
        │   └─ Lee /img/ → Redimensiona → Genera máscaras
        │   └─ Salida: /imagenes_procesadas/ (images, masks, masked_images)
        │
┌─────────────────────────────────────────────────────────────┐
│ FASE 2: ENTRENAMIENTO DEL MODELO DE DIFUSIÓN INVERSA      │
└─────────────────────────────────────────────────────────────┘
        │
        ├─→ dataset.py
        │   └─ Carga pares (dañada, original) desde /imagenes_procesadas/
        │   └─ Normaliza y convierte a tensores
        │
        ├─→ diffusion.py
        │   └─ add_noise() corrompe imágenes con ruido gaussiano
        │
        ├─→ unet.py
        │   └─ Define arquitectura U-Net para predicción de ruido
        │
        ├─→ train_diffusion.py (EJECUTAR ESTO)
        │   ├─ Instancia Dataset + DataLoader
        │   ├─ Carga modelo U-Net
        │   ├─ Loop de entrenamiento (5 epochs):
        │   │  ├─ Add ruido a imagen original
        │   │  ├─ Predice ruido con U-Net
        │   │  ├─ Calcula loss (MSE)
        │   │  ├─ Optimiza pesos
        │   │  └─ Visualiza reconstrucción
        │   └─ Salida: Modelo entrenado + gráficas
        │
        ├─→ visualize.py
        │   └─ Muestra comparación en tiempo real
        │
┌─────────────────────────────────────────────────────────────┐
│ RESULTADO FINAL                                             │
└─────────────────────────────────────────────────────────────┘
    └─ Modelo U-Net capaz de predecir ruido
    └─ Usado para reconstruir áreas dañadas (reverse diffusion)
    └─ Listo para integración con M2 (GAN de bordes) y Swin Transformer
```

---

## 🚀 INSTRUCCIONES DE EJECUCIÓN PASO A PASO

### Requisitos previos
```bash
pip install torch torchvision torchaudio
pip install opencv-python matplotlib numpy pandas requests beautifulsoup4
```

### Ejecución completa

**Paso 1: Descargar imágenes**
```bash
python descargaImagenes.py
# Salida: /img/ con ~200 imágenes de cerámica
# Duración: ~5-10 minutos (depende conexión)
```

**Paso 2: Preprocesar imágenes**
```bash
python preprocesamiento.py
# Salida: /imagenes_procesadas/ con subdirectorios
# - images/ (cerámica original 256x256)
# - masks/ (máscaras de daño)
# - masked_images/ (cerámica CON daño visible)
# Duración: ~1-2 minutos
```

**Paso 3: Entrenar modelo de difusión**
```bash
cd training
python train_diffusion.py
# Muestra:
#   - Loss por epoch
#   - Gráficas de reconstrucción en vivo
# Duración: ~10-20 minutos (GPU recomendada)
```

---

## 📊 Salida esperada

Tras ejecutar `train_diffusion.py`:

```
Epoch 1/5 - Loss: 0.156234
[Gráfica 1: dañada | original | reconstrucción]

Epoch 2/5 - Loss: 0.082145
[Gráfica 2: dañada | original | reconstrucción]

... (3-5 gráficas más)
```

El loss debe **disminuir** en cada epoch (convergencia).

---

## 🔗 Integración futura

Este módulo (M3) es compatible con:
- **M2 (GAN):** Proporciona guía de bordes → `edge_guidance` en diffusion.py
- **M4 (Swin Transformer):** Hook preparado para atención local en p_sample()

Para integración, modificar `train_diffusion.py`:
```python
# Cargar salida de M2 (bordes)
edge_guidance = gan_model(masked)

# Entrenar con guía
reverse_diffusion(..., edge_guidance=edge_guidance, ...)
```