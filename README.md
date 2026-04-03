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
    python proprocesamiento.py
```

Con los comandos anteriores obtendremos la selección de imagenes que usaremos como referencias, y a esa selección aplicada una mascára binaria en la que aplicaremos nuestros algoritmos