```
     █████╗ ███████╗██████╗ ██████╗ ███████╗
    ██╔══██╗██╔════╝██╔══██╗██╔══██╗██╔════╝
    ███████║█████╗  ██████╔╝██████╔╝███████╗
    ██╔══██║██╔══╝  ██╔═══╝ ██╔══██╗╚════██║
    ██║  ██║██║     ██║     ██║  ██║███████║
    ╚═╝  ╚═╝╚═╝     ╚═╝     ╚═╝  ╚═╝╚══════╝
    ═══════════════════════════════════════════
     Advanced Facial & Person Recognition System
    ═══════════════════════════════════════════
```

# 🚀 AFPRS - Sistema Avanzado de Reconocimiento Facial y de Personas

[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-RTX_5060-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://www.nvidia.com/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![YOLO](https://img.shields.io/badge/YOLO-v11-00FFFF?style=for-the-badge)](https://ultralytics.com/)

> **Sistema profesional de visión por computadora** optimizado para RTX 5060 8GB + Intel Core Ultra 7 240H

---

## 📑 Tabla de Contenidos

- [🎯 Visión del Proyecto](#-visión-del-proyecto)
- [🐳 Arquitectura](#-arquitectura-docker)
- [📁 Estructura del Proyecto](#-estructura-del-proyecto)
- [⚡ Requisitos Previos](#-requisitos-previos)
- [🚀 Instalación](#-instalación)
- [🖥️ Configuración VcXsrv](#️-configuración-vcxsrv-servidor-x11)
- [▶️ Ejecución](#️-ejecución)
- [🎮 Controles](#-controles)
- [⚙️ Configuración](#️-configuración)
- [📊 Uso de Recursos](#-uso-de-recursos)
- [🔧 API del Servicio LLM](#-api-del-servicio-llm)
- [🐛 Solución de Problemas](#-solución-de-problemas)
- [📝 Licencia](#-licencia)

---

## 🎯 Visión del Proyecto

Sistema profesional de **reconocimiento facial + detección de personas** que:

| Capacidad | Descripción |
|-----------|-------------|
| 👤 **Detección de Personas** | YOLO11 con pose estimation (17 keypoints del cuerpo) |
| 😊 **Análisis Facial** | 468 landmarks faciales + tracking de iris en tiempo real |
| 🤖 **Descripciones IA** | Generación automática con LLM local (Qwen2.5-1.5B) |
| 🐳 **Arquitectura Docker** | 2 contenedores Linux aislados y optimizados |
| 🎨 **Interfaz Técnica** | HUD minimalista con visualización de toda la tecnología |

---

## 🐳 Arquitectura Docker

```
┌─────────────────────────────────────────────────────────────────┐
│                        HOST (Windows 11)                         │
│                   RTX 5060 8GB + Core Ultra 7 240H              │
└─────────────────────────────────────────────────────────────────┘
                              │
                    Docker Desktop + WSL2
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                           ▼
┌───────────────────────┐               ┌───────────────────────┐
│   CONTAINER 1         │               │   CONTAINER 2         │
│   afprs-vision        │    HTTP/WS    │   afprs-llm           │
│                       │◄─────────────►│                       │
│ • YOLO11 (person)     │   Port 8001   │ • Qwen2.5-1.5B        │
│ • YOLO11 (face)       │               │ • Generación texto    │
│ • MediaPipe FaceMesh  │               │ • API FastAPI         │
│ • ByteTrack           │               │ • 4-bit quantization  │
│ • OpenCV + Display    │               │                       │
│                       │               │                       │
│ Puerto: 8000          │               │ Puerto: 8001          │
│ GPU: ~5GB VRAM        │               │ GPU: ~2GB VRAM        │
└───────────────────────┘               └───────────────────────┘
```

---

## 📁 ESTRUCTURA DEL PROYECTO

```
ADVANCED-FACIAL-PERSON-RECOGNITION-SYSTEM-AFPRS-OPEN-MODEL/
├── docker-compose.yml
├── .env
├── README.md
│
├── vision/                          # Container 1: Vision
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py                  # Entry point
│   │   ├── config.py                # Configuración
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── capture.py           # Captura video
│   │   │   ├── person_detector.py   # YOLO11 personas
│   │   │   ├── face_detector.py     # YOLO11 rostros
│   │   │   ├── associator.py        # Asociación face↔person
│   │   │   ├── tracker.py           # ByteTrack
│   │   │   ├── face_analyzer.py     # MediaPipe 468pts
│   │   │   ├── person_analyzer.py   # Análisis corporal
│   │   │   └── llm_client.py        # Cliente HTTP al LLM
│   │   ├── display/
│   │   │   ├── __init__.py
│   │   │   ├── overlays.py          # Bboxes técnicos
│   │   │   ├── skeleton.py          # Visualización pose
│   │   │   ├── face_mesh.py         # Landmarks faciales
│   │   │   ├── hud.py               # Métricas tiempo real
│   │   │   └── panel.py             # Panel descripciones
│   │   └── utils/
│   │       └── __init__.py
│   └── models/                      # Modelos YOLO
│
├── llm/                             # Container 2: LLM
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── main.py                  # FastAPI server
│       ├── config.py
│       └── generator.py             # Generador descripciones
│
└── scripts/
    ├── build.sh / build.ps1
    ├── run.sh / run.ps1
    └── download_models.sh / download_models.ps1
```

---

## 📋 Características Principales

### 🔍 Pipeline de Visión

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Captura   │───►│  Detección  │───►│  Tracking   │───►│  Análisis   │
│   Video     │    │ YOLO11 x2   │    │  ByteTrack  │    │  MediaPipe  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                │
                   ┌─────────────┐    ┌─────────────┐           │
                   │   Display   │◄───│     LLM     │◄──────────┘
                   │   OpenCV    │    │  Qwen2.5    │
                   └─────────────┘    └─────────────┘
```

### 📊 Tecnologías Integradas

| Módulo | Tecnología | Función |
|--------|------------|--------|
| 👤 **Detección Personas** | YOLO11m-pose | 17 keypoints del esqueleto humano |
| 😊 **Detección Rostros** | YOLO11n-face | Bounding boxes faciales de alta precisión |
| 🎭 **Malla Facial** | MediaPipe FaceMesh | 468 landmarks + 10 puntos de iris |
| 🔄 **Seguimiento** | ByteTrack | IDs persistentes entre frames |
| 👕 **Análisis Ropa** | OpenCV HSV | Clasificación de colores dominantes |
| 🤖 **Descripciones** | Qwen2.5-1.5B | Generación de texto con LLM local |
| 🐳 **Contenedores** | Docker + NVIDIA | Aislamiento y aceleración GPU |

---

## ⚡ Requisitos Previos

Antes de instalar AFPRS, asegúrate de tener:

### Hardware Mínimo
- **GPU NVIDIA**: 6GB+ VRAM (Recomendado: RTX 3060 o superior)
- **RAM**: 16GB mínimo
- **CPU**: Intel Core i5 / AMD Ryzen 5 o superior
- **Webcam**: USB o integrada

### Software Requerido

| Software | Versión | Propósito |
|----------|---------|----------|
| Windows 11 | 22H2+ | Sistema operativo host |
| Docker Desktop | 4.0+ | Contenedores Linux |
| WSL2 | Ubuntu 22.04 | Subsistema Linux |
| NVIDIA Driver | 525+ | Drivers GPU |
| NVIDIA Container Toolkit | Latest | GPU en Docker |
| VcXsrv | 1.20+ | Servidor X11 para GUI |

### Verificar Requisitos

```powershell
# Verificar Docker
docker --version

# Verificar WSL2
wsl --list --verbose

# Verificar GPU NVIDIA
nvidia-smi

# Verificar NVIDIA Container Toolkit (Docker debe estar corriendo)
docker run --rm --gpus all nvidia/cuda:12.4-base-ubuntu22.04 nvidia-smi
```

---

## 🎮 Controles de la Interfaz

| Tecla | Acción |
|-------|--------|
| `q` | Salir |
| `p` | Toggle person bbox |
| `f` | Toggle face bbox |
| `s` | Toggle skeleton |
| `m` | Toggle face mesh |
| `t` | Toggle trajectory |

---

## 🚀 Instalación

### Paso 1: Clonar el Repositorio

```powershell
# Navegar a tu directorio de proyectos
cd E:\

# El proyecto ya debería estar creado en:
cd ADVANCED-FACIAL-PERSON-RECOGNITION-SYSTEM-AFPRS-OPEN-MODEL
```

### Paso 2: Descargar Modelos YOLO

```powershell
# Windows PowerShell
.\scripts\download_models.ps1
```

Esto descargará:
- `yolo11m-pose.pt` (~50MB) - Detección de personas con pose
- `yolo11n-face.pt` (~6MB) - Detección de rostros

### Paso 3: Construir Contenedores Docker

```powershell
# Asegúrate de que Docker Desktop esté corriendo
.\scripts\build.ps1
```

⏱️ **Tiempo estimado**: 5-15 minutos (primera vez)

---

## 🖥️ Configuración VcXsrv (Servidor X11)

VcXsrv permite mostrar la interfaz gráfica de OpenCV (que corre en Linux/Docker) en tu escritorio Windows.

### ¿Por qué es necesario?

```
┌─────────────────────────────────────────────────────────────┐
│  Container Docker (Linux)                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  cv2.imshow("AFPRS", frame)                           │  │
│  │  → Genera señal X11                                   │  │
│  └─────────────────────────┬─────────────────────────────┘  │
└─────────────────────────────┼───────────────────────────────┘
                              │ Protocolo X11
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Windows (Host)                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  VcXsrv recibe X11 → Dibuja ventana en Windows        │  │
│  └───────────────────────────────────────────────────────┘  │
│                              ▼                              │
│              ┌─────────────────────────────┐                │
│              │    🖼️ Ventana AFPRS         │                │
│              │    Video en tiempo real     │                │
│              │    con todas las            │                │
│              │    visualizaciones          │                │
│              └─────────────────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

### Configurar VcXsrv

1. **Abrir XLaunch** (instalado con VcXsrv)

2. **Pantalla 1 - Display Settings:**
   - ✅ Multiple windows
   - Display number: `0`

3. **Pantalla 2 - Client Startup:**
   - ✅ Start no client

4. **Pantalla 3 - Extra Settings:**
   - ✅ Clipboard
   - ✅ Primary Selection
   - ✅ **Disable access control** ⚠️ (IMPORTANTE)
   - Additional parameters: `-ac`

5. **Pantalla 4 - Finish:**
   - Guardar configuración como `afprs.xlaunch` para uso futuro

### Permitir en Firewall

Cuando Windows pregunte, permitir VcXsrv en:
- ✅ Redes privadas
- ✅ Redes públicas

---

## ▶️ Ejecución

### Iniciar el Sistema

```powershell
# 1. Primero, asegúrate de que VcXsrv esté corriendo
# (Doble clic en tu archivo afprs.xlaunch guardado)

# 2. Iniciar Docker Desktop si no está corriendo
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"

# 3. Esperar a que Docker esté listo (verificar icono en bandeja)

# 4. Ejecutar AFPRS
.\scripts\run.ps1
```

### Primera Ejecución

La primera vez tardará más porque:
1. Descarga la imagen base CUDA (~4GB)
2. Descarga el modelo Qwen2.5-1.5B (~1GB)
3. Inicializa MediaPipe

### Ejecución Normal

```powershell
# Iniciar contenedores
docker-compose up

# Detener contenedores
docker-compose down

# Ver logs en tiempo real
docker-compose logs -f

# Ver logs de un contenedor específico
docker logs afprs-vision -f
docker logs afprs-llm -f
```

---

## ⚙️ Configuración Avanzada

### Variables de Entorno (.env)

```env
# Display X11 para VcXsrv
DISPLAY=:0

# Nombre del proyecto Docker
COMPOSE_PROJECT_NAME=afprs
```

### Configuración del Módulo Vision

Archivo: `vision/app/config.py`

```python
# ═══════════════════════════════════════════════════════════════
# HARDWARE
# ═══════════════════════════════════════════════════════════════
DEVICE: str = "cuda"        # "cuda" para GPU, "cpu" para CPU
USE_FP16: bool = True       # Half precision para mejor rendimiento

# ═══════════════════════════════════════════════════════════════
# CAPTURA DE VIDEO
# ═══════════════════════════════════════════════════════════════
CAMERA_ID: int = 0          # ID de la cámara (0 = default)
FRAME_WIDTH: int = 1920     # Resolución horizontal
FRAME_HEIGHT: int = 1080    # Resolución vertical
TARGET_FPS: int = 60        # FPS objetivo

# ═══════════════════════════════════════════════════════════════
# DETECCIÓN
# ═══════════════════════════════════════════════════════════════
PERSON_CONF: float = 0.5    # Confianza mínima para personas
FACE_CONF: float = 0.5      # Confianza mínima para rostros
PERSON_IMG_SIZE: int = 640  # Tamaño de imagen para inferencia

# ═══════════════════════════════════════════════════════════════
# TRACKING
# ═══════════════════════════════════════════════════════════════
TRACK_BUFFER: int = 30      # Frames antes de eliminar track perdido
MATCH_THRESH: float = 0.8   # Umbral de coincidencia IoU
TRAJECTORY_LEN: int = 60    # Longitud del historial de trayectoria

# ═══════════════════════════════════════════════════════════════
# LLM CLIENT
# ═══════════════════════════════════════════════════════════════
LLM_TIMEOUT: float = 5.0    # Timeout para peticiones al LLM
DESC_INTERVAL: float = 2.5  # Intervalo entre actualizaciones de descripción
```

### Configuración del Módulo LLM

Archivo: `llm/app/config.py`

```python
# Modelo a usar (Hugging Face)
MODEL: str = "Qwen/Qwen2.5-1.5B-Instruct"

# Cuantización 4-bit para reducir VRAM
LOAD_4BIT: bool = True

# Tokens máximos en la respuesta
MAX_TOKENS: int = 80

# Temperatura (0.0 = determinístico, 1.0 = creativo)
TEMPERATURE: float = 0.3
```

---

## 🔧 API del Servicio LLM

El contenedor `afprs-llm` expone una API REST en el puerto `8001`.

### POST /generate

Genera una descripción de texto basada en los atributos de la persona.

**Request:**
```json
{
  "age": 25,
  "gender": "male",
  "emotion": "neutral",
  "gaze": "forward",
  "upper_color": "blue",
  "lower_color": "black",
  "position": "center",
  "depth": "mid"
}
```

**Response:**
```json
{
  "description": "Un hombre joven de unos veinticinco años se encuentra en el centro del encuadre. Viste una parte superior azul con pantalones negros, manteniendo una expresión neutral mientras mira directamente hacia adelante."
}
```

### GET /health

Verifica el estado del servicio.

**Response:**
```json
{
  "status": "ok"
}
```

### Probar la API Manualmente

```powershell
# Health check
curl http://localhost:8001/health

# Generar descripción
curl -X POST http://localhost:8001/generate `
  -H "Content-Type: application/json" `
  -d '{"upper_color": "red", "lower_color": "blue", "position": "left"}'
```

---

## 📊 Uso de Recursos

### Distribución de VRAM (RTX 5060 8GB)

```
┌────────────────────────────────────────────────────────────────┐
│                    VRAM TOTAL: 8GB                             │
├────────────────────────────────────────────────────────────────┤
│ ██████████████████████████░░░░░░░░░░░░░░░░  YOLO11m-pose  2GB  │
│ ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  YOLO11n-face 0.5GB │
│ ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  MediaPipe    0.5GB │
│ ██████████████████████████░░░░░░░░░░░░░░░░  Qwen2.5 4-bit 2GB  │
│ ██████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░  Buffer/Other  1GB  │
├────────────────────────────────────────────────────────────────┤
│ ████████████████████████████████████████████  USADO: ~6GB      │
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  LIBRE: ~2GB      │
└────────────────────────────────────────────────────────────────┘
```

### Tabla de Recursos por Componente

| Componente | Container | VRAM | RAM | CPU |
|------------|-----------|------|-----|-----|
| YOLO11m-pose | vision | ~2.0 GB | ~1.0 GB | Medio |
| YOLO11n-face | vision | ~0.5 GB | ~0.5 GB | Bajo |
| MediaPipe FaceMesh | vision | ~0.5 GB | ~0.5 GB | Bajo |
| OpenCV Display | vision | ~0.1 GB | ~0.2 GB | Bajo |
| Qwen2.5-1.5B (4-bit) | llm | ~2.0 GB | ~2.0 GB | Medio |
| **TOTAL** | - | **~5-6 GB** | **~4 GB** | - |

---

## 🐛 Solución de Problemas

### ❌ Error: "No NVIDIA GPU detected"

**Causa:** Docker no puede acceder a la GPU.

**Solución:**
```powershell
# 1. Verificar que los drivers NVIDIA estén instalados
nvidia-smi

# 2. Verificar NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.4-base-ubuntu22.04 nvidia-smi

# 3. Si falla, reinstalar NVIDIA Container Toolkit:
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### ❌ Error: "Cannot connect to display" / "No display available"

**Causa:** VcXsrv no está corriendo o mal configurado.

**Solución:**
```powershell
# 1. Verificar que VcXsrv esté corriendo (buscar icono X en la bandeja)

# 2. Verificar la variable DISPLAY
echo $env:DISPLAY
# Debe mostrar: host.docker.internal:0.0 o :0

# 3. Reiniciar VcXsrv con "Disable access control" activado

# 4. Verificar firewall - permitir VcXsrv en redes privadas y públicas
```

### ❌ Error: "Camera not found" / "Cannot open video device"

**Causa:** La webcam no está disponible o está siendo usada.

**Solución:**
```powershell
# 1. Cerrar otras apps que usen la cámara (Zoom, Teams, etc.)

# 2. Verificar que la cámara funcione en Windows
# Abrir la app "Cámara" de Windows

# 3. En docker-compose.yml, verificar el dispositivo:
# devices:
#   - /dev/video0:/dev/video0

# 4. Probar con diferente ID de cámara en config.py:
# CAMERA_ID: int = 1  # o 2, etc.
```

### ❌ Error: "Out of memory" / "CUDA out of memory"

**Causa:** No hay suficiente VRAM disponible.

**Solución:**
```powershell
# 1. Cerrar otras aplicaciones que usen GPU

# 2. Reducir resolución en config.py:
FRAME_WIDTH: int = 1280   # en lugar de 1920
FRAME_HEIGHT: int = 720   # en lugar de 1080

# 3. Reducir tamaño de imagen de inferencia:
PERSON_IMG_SIZE: int = 480  # en lugar de 640
```

### ❌ Error: "Connection refused" al LLM

**Causa:** El contenedor LLM no está listo.

**Solución:**
```powershell
# 1. Verificar que ambos contenedores estén corriendo
docker ps

# 2. Ver logs del contenedor LLM
docker logs afprs-llm -f

# 3. Esperar a que aparezca "[LLM] Model ready"

# 4. Verificar conectividad
curl http://localhost:8001/health
```

### 🔄 Reiniciar Todo Desde Cero

```powershell
# Detener todos los contenedores
docker-compose down

# Eliminar imágenes (si hay problemas de build)
docker rmi afprs-vision afprs-llm

# Reconstruir
.\scripts\build.ps1

# Ejecutar
.\scripts\run.ps1
```

---

## 📝 Licencia

```
MIT License

Copyright (c) 2026 AFPRS Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Para contribuir:

1. 🍴 Fork el repositorio
2. 🌿 Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit tus cambios (`git commit -m 'Add: AmazingFeature'`)
4. 📤 Push a la rama (`git push origin feature/AmazingFeature`)
5. 🔄 Abre un Pull Request

---

## 📞 Soporte

Si tienes problemas o preguntas:

1. 📖 Revisa la sección de [Solución de Problemas](#-solución-de-problemas)
2. 🔍 Busca en los issues existentes
3. 📝 Abre un nuevo issue con detalles del problema

---

<div align="center">

```
═══════════════════════════════════════════════════════════════════════════
                    🚀 AFPRS - Advanced Facial & Person Recognition System
                    Optimizado para RTX 5060 8GB + Intel Core Ultra 7 240H
═══════════════════════════════════════════════════════════════════════════
```

**Hecho con ❤️ y mucho ☕**

</div>
