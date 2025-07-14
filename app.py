import gradio as gr
import os
import cv2
import numpy as np

from processing.color_operations import convert_color
from processing.corrections import (
    rotate, flip, resize, adjust_brightness_contrast, gamma_correction, equalize_histogram
)
from processing.enhancements import apply_filter
from processing.masks import (
    apply_threshold, adaptive_threshold, otsu_threshold,
    bitwise_and, bitwise_or, bitwise_not
)
from processing.background_removal import remove_background_hsv, remove_background_lab, grabcut
from processing.background_change import change_background_color, change_background_image
from processing.collage import stack_images
from processing.detection import detect_contours, detect_faces_haar

IMAGE_DIR = "images"
BACKGROUND_DIR = "backgrounds"

def list_images():
    return [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

def list_backgrounds():
    return [f for f in os.listdir(BACKGROUND_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

def load_image(folder, filename):
    path = os.path.join(folder, filename)
    image = cv2.imread(path)
    return image

def process_all(
    filename, color_space, rotate_angle, flip_mode, brightness, contrast, gamma,
    filter_type, threshold_type, threshold_value,
    bitwise_op, background_removal_method, change_bg_mode,
    bg_color, bg_image_name, collage_mode,
    detect_contours_flag, detect_faces_flag
):
    image = load_image(IMAGE_DIR, filename)
    if image is None:
        return np.zeros((300, 300, 3), dtype=np.uint8)

    # Color conversion
    image = convert_color(image, color_space)

    # Corrections
    image = rotate(image, rotate_angle)
    image = flip(image, flip_mode)
    image = adjust_brightness_contrast(image, brightness, contrast)
    image = gamma_correction(image, gamma)

    # Histogram equalization only if grayscale
    if color_space == "GRAYSCALE":
        image = equalize_histogram(image)

    # Apply filter
    if filter_type != "None":
        image = apply_filter(image, filter_type)

    # Thresholding
    if threshold_type == "Binary":
        image = apply_threshold(image, threshold_value)
    elif threshold_type == "Adaptive":
        image = adaptive_threshold(image)
    elif threshold_type == "Otsu":
        image = otsu_threshold(image)

    # Bitwise operations
    if bitwise_op == "AND":
        image = bitwise_and(image, image)
    elif bitwise_op == "OR":
        image = bitwise_or(image, image)
    elif bitwise_op == "NOT":
        image = bitwise_not(image)

    # Background removal with fixed HSV or LAB ranges
    if background_removal_method == "HSV":
        image = remove_background_hsv(image)  # No params needed, uses fixed range
    elif background_removal_method == "LAB":
        image = remove_background_lab(image)
    elif background_removal_method == "GrabCut":
        image = grabcut(image)

    # Background change
    if change_bg_mode == "Color":
        image = change_background_color(image, bg_color)
    elif change_bg_mode == "Image" and bg_image_name:
        bg_image = load_image(BACKGROUND_DIR, bg_image_name)
        image = change_background_image(image, bg_image)

    # Collage
    if collage_mode == "Horizontal":
        image = stack_images([image, image], axis=1)
    elif collage_mode == "Vertical":
        image = stack_images([image, image], axis=0)

    # Contours detection
    if detect_contours_flag:
        image = detect_contours(image)

    # Face detection with Haar cascades
    if detect_faces_flag:
        image = detect_faces_haar(image)

    # Convert to RGB for Gradio
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        return image

def main_interface():
    image_list = list_images()
    bg_list = list_backgrounds()

    with gr.Blocks() as demo:
        gr.Markdown("## 🖼️ Aplicación Completa de Procesamiento de Imágenes")

        with gr.Row():
            image_selector = gr.Dropdown(label="Imagen base", choices=image_list, value=image_list[0] if image_list else None)
            bg_selector = gr.Dropdown(label="Imagen fondo (para cambio)", choices=bg_list, value=bg_list[0] if bg_list else None)
            color_space = gr.Dropdown(label="Espacio de color", choices=["RGB", "HSV", "LAB", "GRAYSCALE"], value="RGB")

        with gr.Row():
            rotate_angle = gr.Slider(-180, 180, value=0, label="Rotación")
            flip_mode = gr.Radio(["horizontal", "vertical", "both"], label="Modo volteo", value="horizontal")

        with gr.Row():
            brightness = gr.Slider(-100, 100, value=0, label="Brillo")
            contrast = gr.Slider(-100, 100, value=0, label="Contraste")
            gamma = gr.Slider(0.1, 3.0, value=1.0, step=0.1, label="Gamma")

        with gr.Row():
            filter_type = gr.Dropdown(label="Filtro", choices=["None", "blur", "gaussian", "bilateral", "median", "sharpen", "sobel", "laplacian", "canny", "emboss", "custom"], value="median")
            threshold_type = gr.Radio(["None", "Binary", "Adaptive", "Otsu"], label="Tipo de umbral", value="None")
            threshold_value = gr.Slider(0, 255, value=128, label="Valor umbral (binario)")

        with gr.Row():
            bitwise_op = gr.Radio(["None", "AND", "OR", "NOT"], label="Operación bitwise", value="None")

        with gr.Row():
            background_removal_method = gr.Radio(["None", "HSV", "LAB", "GrabCut"], label="Eliminación de fondo", value="None")

        with gr.Row():
            change_bg_mode = gr.Radio(["None", "Color", "Image"], label="Cambio de fondo", value="None")
            bg_color = gr.ColorPicker(label="Color fondo")
            bg_image_name = gr.Dropdown(label="Imagen fondo (cambio)", choices=bg_list, visible=False)

        with gr.Row():
            collage_mode = gr.Radio(["None", "Horizontal", "Vertical"], label="Modo collage", value="None")

        with gr.Row():
            detect_contours_flag = gr.Checkbox(label="Detectar contornos")
            detect_faces_flag = gr.Checkbox(label="Detectar rostros (Haar cascades)")

        output_image = gr.Image(label="Resultado")

        inputs = [
            image_selector, color_space, rotate_angle, flip_mode, brightness, contrast, gamma,
            filter_type, threshold_type, threshold_value, bitwise_op, background_removal_method,
            change_bg_mode, bg_color, bg_image_name, collage_mode,
            detect_contours_flag, detect_faces_flag
        ]

        def toggle_bg_image_selector(change_bg):
            return gr.update(visible=(change_bg=="Image"))

        change_bg_mode.change(fn=toggle_bg_image_selector, inputs=change_bg_mode, outputs=bg_image_name)

        for inp in inputs:
            inp.change(fn=process_all, inputs=inputs, outputs=output_image)

        with gr.Row():
            process_button = gr.Button("Procesar Imagen")
            process_button.click(fn=process_all, inputs=inputs, outputs=output_image)

    return demo

if __name__ == "__main__":
    demo = main_interface()
    demo.launch()
