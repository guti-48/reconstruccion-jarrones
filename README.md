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

### **5. `models/unet.py` — Arquitectura U-Net sensible al tiempo para denoisificación**
- **Propósito:** Red neuronal que predice ruido gaussiano condicionada al timestep de difusión
- **Clase:** `UNet(in_channels=8, out_channels=3, time_dim=256)`
- **Entrada:** 
  - Imagen ruidosa + contexto (8 canales = 3 imagen + 3 masked + 1 bordes + 1 máscara)
  - Tensor de timesteps t (posición en el schedule de difusión)
- **Salida:** 
  - Predicción del ruido gaussiano (3 canales, misma resolución que entrada)

#### **Componentes principales:**

**1. TimeEmbedding — Codificación sinusoidal del tiempo**
- Convierte timestep escalar `t` en vector denso de dimensión `time_dim=256`
- Usa posicional encoding sinusoidal (técnica de Transformers):
  $$\text{emb}_{2i} = \sin(t / 10000^{2i/d}), \quad \text{emb}_{2i+1} = \cos(t / 10000^{2i/d})$$
- Pasa por MLP: `Linear(256) → ReLU → Linear(256)`
- **Razón:** Permite que la red sepa en qué paso del proceso de difusión está

**2. ResBlock — Bloque residual con inyección de tiempo**
- **Estructura:**
  ```
  Input → GroupNorm(8) → SiLU → Conv(in→out) 
          ↓ + time_embedding (inyectado espacialmente)
          GroupNorm(8) → SiLU → Conv(out→out) → Output
          ↑___________________________skip connection__↑
  ```
- **Características:**
  - **GroupNorm(8):** Normalización por grupo (más estable que BatchNorm)
  - **SiLU:** Activación suave (mejor que ReLU para difusión)
  - **Time injection:** Suma el embedding del tiempo espacialmente: `h = h + time_mlp(t_emb)[:, :, None, None]`
  - **Skip connection:** Residual convencional para gradientes más estables
- **Ventaja:** La información del timestep se difunde por toda la arquitectura

**3. Down — Bloque de downsampling (encoder)**
- Aplica `ResBlock` + submuestreo espacial
- **Submuestreo:** `Conv2d(kernel=4, stride=2, padding=1)` (mejor que MaxPool)
- **Retorna:** (salida para skip connection, salida downsampled para siguiente nivel)

**4. Up — Bloque de upsampling (decoder)**
- Upsampling con `ConvTranspose2d(kernel=4, stride=2, padding=1)`
- Concatena con skip connection del encoder: `torch.cat([x, skip], dim=1)`
- Aplica `ResBlock` sobre la concatenación
- **Manejo de tamaños:** Interpolación si shapes no coinciden

#### **Arquitectura completa (3 niveles):**
```
Input (B, 8, 256, 256)
  ↓
Down1: ResBlock(8→64) + Conv(stride=2)  → salida (B, 64, 128, 128) [guarda s1]
  ↓
Down2: ResBlock(64→128) + Conv(stride=2) → salida (B, 128, 64, 64) [guarda s2]
  ↓
Down3: ResBlock(128→256) + Conv(stride=2) → salida (B, 256, 32, 32) [guarda s3]
  ↓
Bottleneck: ResBlock(256→256)  [PUNTO MÁS COMPRIMIDO]
  ↓
Up1: ConvTranspose(256→128) + ResBlock(128+256=384→128) → (B, 128, 64, 64) + s3
  ↓
Up2: ConvTranspose(128→64) + ResBlock(64+128=192→64) → (B, 64, 128, 128) + s2
  ↓
Up3: ConvTranspose(64→64) + ResBlock(64+64=128→64) → (B, 64, 256, 256) + s1
  ↓
Output: Conv1x1(64→3) → (B, 3, 256, 256)
```

#### **Detalles de implementación:**
- **Parámetros del constructor:**
  - `in_channels=8`: Concatenación de imagen ruidosa (3) + imagen intacta (3) + bordes (1) + máscara (1)
  - `out_channels=3`: Predice ruido gaussiano de 3 canales (RGB)
  - `time_dim=256`: Dimensión del embedding temporal
- **Skip connections:** Todas las salidas de Down se concatenan en Up (información multiresolución)
- **Activación:** SiLU en lugar de ReLU (suavidad crucial para denoisificación)
- **GroupNorm:** 8 grupos por capa (normalización sin dependencia del batch)

#### **Funcionamiento durante el entrenamiento (forward pass):**
1. Recibe: imagen ruidosa, imagen original, bordes, máscara, timestep t
2. Codifica t con TimeEmbedding
3. Encoder extrae features jerárquicas (64→128→256 canales)
4. Bottleneck actúa como cuello de botella (comprensión máxima)
5. Decoder reconstr uye con skip connections (4x, 8x, 16x tamaños)
6. Output: predice ε_t (ruido a remover)
7. **Loss:** MSE entre ruido predicho y ruido real añadido en forward diffusion

- **Uso:** Entrenar con `train_diffusion.py` para que aprenda a predecir ruido en cualquier timestep
- **No se ejecuta directamente**, se carga como modelo en scripts de entrenamiento e inferencia

### **6. `models/diffusion.py` — Procesos de difusión (Forward + Reverse)**
- **Propósito:** Implementar el pipeline completo de difusión probabilística (DDPM) para generación con inpainting

#### **Clase: `Diffusion`**

**Parámetros de inicialización:**
- `T = 700`: Número total de pasos de difusión (tiempo máximo del proceso)
- `beta_start = 1e-4`, `beta_end = 0.02`: Define el ruido schedule lineal
- `device`: CPU o GPU donde se ejecuta

**Componentes principales del constructor:**

1. **Noise Schedule (β_t):**
   - Secuencia lineal: $\beta_t = \beta_{start} + \frac{t}{T}(\beta_{end} - \beta_{start})$
   - Define cuánto ruido se añade en cada paso (crece gradualmente de 1e-4 a 0.02)
   - Almacenado en `self.beta` (tensor de 700 valores)

2. **Alpha (α_t) y Alpha-hat (ᾱ_t):**
   - $\alpha_t = 1 - \beta_t$ (señal conservada en cada paso)
   - $\bar{\alpha}_t = \prod_{i=1}^{t} \alpha_i$ (producto acumulado, permite saltar directamente a cualquier t)
   - Matemáticamente: $\bar{\alpha}_t$ responde: "¿Cuánta señal queda tras t pasos de ruido?"

**Método 1: `add_noise(x0, t)` — FORWARD DIFFUSION**
- Implementa el proceso de corrupción progresiva: $q(x_t|x_0)$
- **Ecuación DDPM:** 
  $$x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1-\bar{\alpha}_t} \cdot \epsilon$$
  donde $\epsilon \sim \mathcal{N}(0, I)$ es ruido gaussiano puro
- **Entrada:** 
  - `x0`: tensor de imagen original (B, 3, 256, 256)
  - `t`: tensor de timesteps (B,) con valores en [0, T)
- **Salida:** tupla `(x_t, noise)`
  - `x_t`: imagen ruidosa en el paso t
  - `noise`: el ruido gaussiano utilizado (para entrenar con MSE Loss)
- **Uso:** Generar datos de entrenamiento corrompidos para el modelo denoisificador

**Método 2: `sample(model, masked, edges, mask)` — REVERSE DIFFUSION con REPAINT**
- Implementa la reconstrucción iterativa desde ruido puro hasta imagen limpia
- Incluye **REPAINT inpainting**: mantiene forzadas las regiones intactas durante la decodificación
- **Parámetros:**
  - `model`: Red U-Net entrenada para predecir ruido
  - `masked`: Imagen dañada (con huecos) - (B, 3, 256, 256)
  - `edges`: Bordes estructurales extraídos - (B, 1, 256, 256)
  - `mask`: Máscara binaria (1=daño/hueco, 0=intacto) - (B, 1, 256, 256)
- **Algoritmo iterativo (T → 1):**
  1. **Inicializa** con ruido puro: $x_T \sim \mathcal{N}(0, I)$
  2. **Loop inverso** para cada t en [T-1, ..., 1]:
     - Concatena entrada: $[x_t, masked, edges, mask]$ y pasa al modelo
     - Red predice: $\hat{\epsilon}_t = \text{model}([x_t, masked, edges, mask], t)$
     - **Denoisificación normal:** 
       $$x_{pred} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \hat{\epsilon}_t \right) + \sqrt{\beta_t} \cdot z$$
       donde $z \sim \mathcal{N}(0, I)$ para $t > 1$, $z = 0$ para $t = 1$
     - **REPAINT Intervention** (clave para inpainting):
       - Calcula cómo se vería la región intacta/conocida ruidosa en el paso anterior:
         $$x_{known,t-1} = \sqrt{\bar{\alpha}_{t-1}} \cdot masked + \sqrt{1-\bar{\alpha}_{t-1}} \cdot \epsilon$$
       - **Fusión híbrida:** Fuerza las regiones conocidas y permite imaginación en daños:
         $$x_{t-1} = (x_{pred} \cdot mask) + (x_{known,t-1} \cdot (1-mask))$$
     - Esto garantiza que las zonas intactas nunca se desvíen de la realidad
  3. **Retorna:** imagen reconstruida clipeada a [0, 1]

- **Uso:** Generar predicción de regiones dañadas mientras se preservan áreas intactas
- **No se ejecuta directamente**, se importa y usa en `train_diffusion.py`

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


### **10. `training/train_diffusion.py` — Script de entrenamiento de difusión inversa**
- **Propósito:** Entrenar el modelo U-Net para predecir ruido en el proceso de reverse diffusion
- **Parámetros principales:**
  - `T=200`: Número de pasos de difusión (puede subirse a 500 para mejor calidad)
  - `EPOCHS=10`: Número de epochs de entrenamiento
  - `BATCH_SIZE=4`: Tamaño del batch
  - `LR=1e-4`: Learning rate del optimizer Adam
  - `device`: GPU/CPU automáticamente detectado

#### **Flujo de entrenamiento (Loop principal):**

**Paso 1: Preparación de datos**
```
1. Crear Dataset y DataLoader desde data/processed/
2. Instanciar:
   - Diffusion(T=200): Clase para forward/reverse diffusion
   - UNet(in_channels=8, out_channels=3): Red denoisificadora
   - Optimizer: Adam(lr=1e-4)
3. Inicializar best_loss = inf para checkpoint management
```

**Paso 2: Bucle de entrenamiento (10 epochs)**
```
Para cada epoch:
  Para cada batch (masked, real, gray, edges, mask):
    
    A. SAMPLING DE TIMESTEP:
       - t = randint(0, T=200) → timestep aleatorio en [0, 200)
       - Cada imagen del batch puede tener t diferente
    
    B. FORWARD DIFFUSION:
       - x_t, noise = diffusion.add_noise(real, t)
       - Corrompe la imagen original con ruido en paso t
       - noise es el ruido gaussiano real (N(0,I))
    
    C. PREPARACIÓN DE ENTRADA A LA RED:
       - input_model = [x_t (3) + masked (3) + edges (1) + mask (1)]
       - Total: 8 canales de información contextual
    
    D. PREDICCIÓN DE RUIDO:
       - predicted_noise = model(input_model, t)
       - Red predice qué ruido se añadió
    
    E. CÁLCULO DE LOSS (IMPORTANTE - WEIGHTED):
       - loss = ((noise - predicted_noise)² * mask).sum() / (mask.sum() + ε)
       - Se calcula SOLO en la región dañada (mask=1)
       - Ignore el resto del plato (mask=0)
       - Evita que la red "olvide" copiar áreas intactas
    
    F. BACKPROPAGATION Y ACTUALIZACIÓN:
       - optimizer.zero_grad()
       - loss.backward()
       - optimizer.step()
    
  Imprime: "Epoch X/10 - Loss: Y.YYYY"
  
  Si loss mejora (< best_loss):
    - Guarda checkpoint: checkpoints/diffusion_best.pth
    - Imprime: "Modelo guardado (mejor hasta ahora)"
```

**Paso 3: Convergencia esperada**
- El loss debe **disminuir** monótonamente a lo largo de los epochs
- Típicamente: 0.5 → 0.3 → 0.2 → 0.15 → ...
- Si loss sube o no converge: revisar learning rate, architecture, o data quality

#### **Diferencias con DDPM clásico:**
| Aspecto | DDPM Clásico | Este Modelo |
|--------|-------------|-----------|
| **Pérdida de datos** | Predice ruido en imagen entera | Predice ruido SOLO en agujeros |
| **Contexto** | Solo imagen ruidosa | Imagen + original + bordes + máscara |
| **T (pasos)** | Típicamente 1000 | Aquí 200 (más rápido) |
| **Aplicación** | Generación de texto-a-imagen | Inpainting de cerámica |

#### **Archivos generados:**
- `checkpoints/diffusion_best.pth`: Pesos del modelo con menor loss
- Histórico de losses en console (registrado en logs si se habilita)

- **Ejecución:**
  ```bash
  cd training
  python train_diffusion.py
  # Duración: ~10-30 minutos en GPU, ~2-3 horas en CPU
  ```

- **Salida esperada:**
  ```
  Entrenando Diffusion (fase final)...
  
  Epoch 1/10 - Loss: 0.4567
  Modelo guardado (mejor hasta ahora)
  Epoch 2/10 - Loss: 0.3421
  Modelo guardado (mejor hasta ahora)
  Epoch 3/10 - Loss: 0.2891
  ...
  Epoch 10/10 - Loss: 0.1234
  
  Entrenamiento Diffusion completado
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
