# Aplicación Interactiva de Procesamiento de Imágenes

Esta aplicación permite procesar imágenes aplicando diferentes transformaciones, filtros, correcciones de color, eliminación de fondo y más, usando **OpenCV**, **NumPy**, y una interfaz interactiva basada en **Gradio**.

## Requisitos

- Python 3.8 o superior
- OpenCV
- NumPy
- Gradio 3.48Q

Instalación:

```bash
pip install -r requirements.txt

image_processing_app/
├── app.py                  # Interfaz Gradio e integración de módulos
├── images/                  # Carpeta con imágenes de entrada
├── processing/              # Módulos de procesamiento
│   ├── color_operations.py
│   ├── corrections.py
│   ├── enhancements.py
│   ├── masks.py
│   ├── background_removal.py
│   ├── collage.py
│   ├── utils.py
│   └── __init__.py
├── requirements.txt
└── README.md
