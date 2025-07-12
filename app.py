import gradio as gr
import os
import cv2
import numpy as np

from processing.color_operations import convert_color
from processing.corrections import rotate, flip, resize, adjust_brightness_contrast, gamma_correction, equalize_histogram
from processing.enhancements import apply_filter
from processing.masks import apply_threshold, adaptive_threshold, otsu_threshold
from processing.background_removal import remove_background_hsv, remove_background_lab, grabcut
from processing.background_change import change_background
from processing.collage import stack_images
from processing.utils import to_bgr

# Ruta de las imágenes
IMAGE_DIR = "images"

def list_images():
    return [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

def load_image(filename):
    path = os.path.join(IMAGE_DIR, filename)
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {filename}")
    return image

def process_all(filename, color_space, rotate_angle, flip_mode, brightness, contrast, gamma,
                filter_type, threshold_value, bg_mode, bg_color, bg_image_filename):
    image = load_image(filename)

    # Convertir a BGR si es gris para evitar errores en funciones que esperan 3 canales
    image = to_bgr(image)

    # Conversión de espacio de color
    image = convert_color(image, color_space)

    # Corrección geométrica y tono
    image = rotate(image, rotate_angle)
    image = flip(image, flip_mode)
    image = adjust_brightness_contrast(image, brightness, contrast)
    image = gamma_correction(image, gamma)

    # Aplicar filtro
    image = apply_filter(image, filter_type)

    # Generar máscara para cambio de fondo (usar umbral sobre escala de grises)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = apply_threshold(gray, threshold_value)

    # Aplicar cambio de fondo según opción
    if bg_mode == "color":
        # Convertir color hex a BGR tupla
        color_bgr = tuple(int(bg_color[i:i+2], 16) for i in (5, 3, 1))
        result = change_background(image, mask, color=color_bgr)
    elif bg_mode == "image" and bg_image_filename:
        bg_img = load_image(bg_image_filename)
        result = change_background(image, mask, new_background=bg_img)
    else:
        result = image

    # Convertir BGR a RGB para mostrar en Gradio
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result_rgb

def main_interface():
    image_list = list_images()
    bg_image_list = list_images()  # Podrías usar otra carpeta para fondos si quieres

    with gr.Blocks() as demo:
        gr.Markdown("# Aplicación Interactiva de Edición de Fotos")

        with gr.Row():
            with gr.Column(scale=1):
                image_selector = gr.Dropdown(label="Selecciona una imagen", choices=image_list)
                color_space = gr.Dropdown(label="Espacio de color", choices=["RGB", "HSV", "LAB", "GRAYSCALE"], value="RGB")
                rotate_angle = gr.Slider(-180, 180, value=0, label="Ángulo de rotación")
                flip_mode = gr.Radio([-1, 0, 1], label="Modo Flip (-1=ambos, 0=vertical, 1=horizontal)", value=1)
                brightness = gr.Slider(-100, 100, value=0, label="Brillo")
                contrast = gr.Slider(-100, 100, value=0, label="Contraste")
                gamma = gr.Slider(0.1, 3.0, value=1.0, step=0.1, label="Gamma")
                filter_type = gr.Dropdown(label="Filtro", choices=["blur", "gaussian", "bilateral", "median", "sharpen", "sobel", "laplacian", "canny", "emboss", "custom"], value="median")
                threshold_value = gr.Slider(0, 255, value=128, label="Umbral binario")

                bg_mode = gr.Radio(["none", "color", "image"], label="Modo cambio de fondo", value="none")
                bg_color = gr.ColorPicker(value="#00ff00", label="Color de fondo")
                bg_image_selector = gr.Dropdown(label="Imagen de fondo", choices=bg_image_list)

                process_button = gr.Button("Procesar Imagen")

            with gr.Column(scale=2):
                output_image = gr.Image(label="Resultado", interactive=False)

        process_button.click(
            process_all,
            inputs=[image_selector, color_space, rotate_angle, flip_mode, brightness, contrast, gamma,
                    filter_type, threshold_value, bg_mode, bg_color, bg_image_selector],
            outputs=output_image
        )

    return demo

if __name__ == "__main__":
    demo = main_interface()
    demo.launch()
