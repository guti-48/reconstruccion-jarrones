# Predicción Faltante de Áreas en vasijas de Cerámica

Este repositorio contiene la implementació del Trabajo Dirigido de Procesamiento de Imagenes Digitales. El objetivo del proyecto es reconstruir digitalmente las áreas faltantes, ya sea roturas o desgastes, en imágenes 2D de procelana antigua utilzando un enfoque dual: Predicción de Patrones (Deep Adversial) y Predicción de color/textura (Reverser Diffusion).

Este proyecto esta basado en el paper "Digital prediction of ancient ceramic images missing areas based on deep adversarial and reverse diffusion" 

## Estructura del proyecto

Nos moveremos a lo laro de varias entregas, tendremos que añadir la primera

El desarrollo de esta segunda estapa está dividido en tres módulos principales, correspondientes a las responsabilidades del equipo:

- **Módulo 1: Datos y Procesamiento** Extracción, filtrado y generacion de datos sintéticos
- **Módulo 2: Estrcutura**  Generación de bordes mediante `edge-connection`
- **Módulo 3: Textura**  Relleno de color mediante la difusión inversa con `repaint`



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

## Módulos Python y su función

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
- **Propósito:** Interfaz para acceder a las imágenes durante el entrenamiento y extraer los bordes estructurales
- **Clase:** `CeramicDataset`
- **Funcionalidad:**
  - Lee pares (imagen_dañada, imagen_original)
  - Normaliza a valores [0, 1]
  - Convierte a tensores PyTorch (B, 3, 256, 256)
  - Compatible con `DataLoader` para batches
  - Canny Adaptativo: Calcula dinámicamente los umbrales basados en la mediana de los píxeles para extraer los bordes reales de la cerámica, suprimiendo ruido
  - Retorno: Devuelve una tupla de 3 tensores: (masked_tensor, img_tensor, edges_tensor)
- **No se ejecuta directamente**, se importa en los scripts de validación y entrenamiento

### **4. `utils/networks.py` — Arquitectura Generativa Adversaria (GAN)**
- **Propósito:** Definir la estructura base del modelo de predicción de patrones basado en la arquitectura de edge-connect
- **Clase:** `EdgeGenerator(x)` y `Discriminator(x)`
  - Implementa submuestreo espacial: El codigo usa `MaxPool2d` en lugar de convoluciones con salto
  - Implementa estabilización: Normalización Espectral en todas las capas convolucionales
  - Evalúa textura: El Discriminador usa arquitectura *PatchGAN*
  - **Entrada:** tensor imagen en gris + tensor bordes + máscara binaria
  - **Salida:** tensor con la predicción de la estructura de líneas (bordes generados)
- **Uso:** Inferir y reconstruir el esqueleto estructural de los patrones faltantes
- **No se ejecuta directamente**, se carga en `train_networks.py`

### **5. `models/unet.py` — Arquitectura de red neuronal**
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

### **6. `models/diffusion.py` — Funciones de difusión**
- **Propósito:** Implementar forward diffusion (corrupción con ruido)
- **Función principal:** `add_noise(x, t)`
  - Implementa ecuación DDPM: $x_t = \sqrt{\alpha_t} \cdot x_0 + \sqrt{1-\alpha_t} \cdot \epsilon$
  - `α_t = 0.9^t` (decae exponencialmente)
  - `ε ~ N(0,1)` (ruido gaussiano)
  - **Entrada:** imagen original + timestep t
  - **Salida:** imagen ruidosa + ruido gaussiano
- **Uso:** Generar datos de entrenamiento corrompidos
- **No se ejecuta directamente**, se importa en `train_diffusion.py`

### **7. `utils/visualize.py` — Visualización de resultados**
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

### **8. `test/test_dataset.py` Validación de Datos y bordes**
- **Propósito:** Comprobacion visual de que el Dataloader carga correctamente las imagenes y extrae los bordes estructurales antes de entrenear a la IA.
- **Flujo de ejecución:** 
  ```
  1. Instanciar CeramicDataset apuntando a processed/
  2. Extraer un elemento (índice 0) del dataset
  3. Convertir tensores PyTorch a arrays Numpy para visualización
  4. Generar plot de Matplotlib con 5 sub-gráficas:
     a) Ground Truth (Intacta)
     b) Cerámica Rota (Masked)
     c) Escala de Grises
     d) Canny Adaptativo (Bordes extraídos dinámicamente)
     e) Máscara Generada
  5. Mostrar la ventana gráfica interactiva
  ```
- **Parámetros:**
  - `root_dir=../processed`
  - `sigma=0.33`
  - `figsize=(20, 3)`
- **Ejecución:** 
```bash
  cd training
  python train_dataset.py
  ```

### **9. `test/test_networks.py` — Validación de Arquitectura**
- **Propósito:** Ejecutar una simulacion para asegurar que no hay errores dimensionales tras adaptar el Maximum Pooling en el Generador.
- **Flujo de ejecución:** 
  ```
  1. Crear tensores "dummy" (aleatorios) simulando imágenes reales
  2. Instanciar EdgeGenerator y Discriminator
  3. Test del Generador (EdgeGenerator):
    a) Pasar tensor de entrada (1, 3, 256, 256)
    b) Validar ejecución sin errores de capas
    c) Comprobar dimensión de salida esperada (1, 1, 256, 256)
  4. Test del Discriminador (PatchGAN):
    a) Pasar tensor de entrada (1, 4, 256, 256)
    b) Validar ejecución sin errores con Normalización Espectral
    c) Comprobar dimensión de la matriz de parches resultante
  5. Imprimir logs de éxito y dimensiones por consola
  ```
- **Parámetros:**
  - `dummy_input_gen=(1, 3, 256, 256)`
  - `dummy_input_disc=(1, 4, 256, 256)`
  - `use_spectral_norm=True`
- **Ejecución:** 
```bash
  cd training
  python train_networks.py
  ```

### **10. `training/train_diffusion.py` — Script de entrenamiento**
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

### **11. `verImagen.py` — Inferencia y Validación Visual Final**
- **Propósito:** Ejecutar la restauración completa de una vasija utilizando los modelos pre-entrenados.
- **Proceso:** Carga automáticamente el mejor checkpoint histórico (`.pth`), extrae los bordes con la GAN, y realiza 200 pasos de *Reverse Diffusion* aplicando el recorte de silueta y la técnica *Repaint*.
- **Salida:** Gráfica comparativa de 4 columnas (Dañada, Bordes, Reconstruida, Original).

---

## FLUJO COMPLETO DEL PROYECTO

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
        │   └─ Extrae bordes estructurales
        │
        ├─→ test_dataset.py
        │   └─ Visualiza la correcta extracción de bordes y máscaras
        │
        ├─→ networks.py
        │   ├─ EdgeGenerator (Codigo con MaxPool + Residuales)
        │   └─ Discriminator (PatchGAN + Normalización Espectral)
        │
        ├─→ test_networks.py
        │   └─ Valida dimensiones de tensores simulados
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
    └─ Modelo de Predicción de Patrones para estructura
    └─ Modelo U-Net capaz de predecir ruido
    └─ Usado para reconstruir áreas dañadas (reverse diffusion)
```

---

## INSTRUCCIONES DE EJECUCIÓN PASO A PASO

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

**Paso 3: Validar extracción de bordes y arquitectura GAN**
```bash
cd test
python test_dataset.py
# Muestra:
#   - Ventana gráfica interactiva con 3 sub-imágenes (Ground Truth, Masked, Canny)
# Duración: ~5-10 segundos
```

```bash
cd test
python test_networks.py
# Muestra:
#   - Verificación matemática por consola del tamaño de los tensores de salida
# Duración: ~5-10 segundos
```

**Paso 4: Entrenar modelo de difusión**
```bash
cd training
python train_diffusion.py
# Muestra:
#   - Loss por epoch
#   - Gráficas de reconstrucción en vivo
# Duración: ~10-20 minutos (GPU recomendada)
```

**Paso 5: Inferencia Rápida (Restauración Visual)**
Si no desea entrenar el modelo desde cero, asegúrese de tener los pesos en la carpeta `/checkpoints/` y ejecute:
```bash
python verImagen.py
---

## Salida esperada
Tras ejecutar `test_dataset.py`:
```
[Gráfica: Ground Truth | Escala de Grises | Canny Adaptativo]
```

Tras ejecutar `test_network.py`:
```
--- Iniciando prueba de la arquitectura de Redes ---
Instanciando EdgeGenerator...
¡Éxito! Forma de salida del Generador: torch.Size([1, 1, 256, 256])
 (Esperado: [1, 1, 256, 256] -> 1 imagen, 1 canal de bordes, 256x256)

Instanciando Discriminador...
¡Éxito! Forma de salida del Discriminador: torch.Size([1, 1, 32, 32])
 (Esperado: Una matriz más pequeña, ej: [1, 1, 32, 32], evaluando los parches)

¡La arquitectura funciona perfectamente y no hay errores de dimensiones!
```

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

## Integración futura

Este módulo (M3) es compatible con:
- **M2 (GAN):** Proporciona guía de bordes → `edge_guidance` en diffusion.py

Para integración, modificar `train_diffusion.py`:
```python
# Cargar salida de M2 (bordes)
edge_guidance = gan_model(masked, edges, mask)

# Entrenar con guía
reverse_diffusion(..., edge_guidance=edge_guidance, ...)
```