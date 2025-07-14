import gradio as gr
import os
import cv2
import numpy as np

# Importaciones de tus módulos de procesamiento
from processing.color_operations import convert_color, hex_to_rgb
from processing.corrections import (
    rotate, flip, adjust_brightness_contrast, gamma_correction, equalize_histogram
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
from processing.utils import to_bgr # Asegúrate de que utils.py tenga esta función

IMAGE_DIR = "images"
BACKGROUND_DIR = "backgrounds"

# Variables globales para almacenar la máscara y la imagen original para la exportación transparente
background_mask_global = None # Máscara donde el objeto es 255, fondo 0
processed_foreground_global = None # Imagen con fondo negro (resultado de la eliminación)
original_image_for_transparent_export_global = None # Almacena la imagen original BGR


def list_images():
    """Lista los archivos de imagen en el directorio de imágenes."""
    return [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

def list_backgrounds():
    """Lista los archivos de imagen en el directorio de fondos."""
    return [f for f in os.listdir(BACKGROUND_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

def load_image(folder, filename):
    """Carga una imagen desde una carpeta específica."""
    path = os.path.join(folder, filename)
    try:
        img = cv2.imread(path)
        if img is None:
            print(f"ADVERTENCIA: No se pudo cargar la imagen desde {path}. ¿Archivo corrupto o formato no soportado?")
        return img
    except Exception as e:
        print(f"ERROR al cargar la imagen {path}: {e}")
        return None


def process_all(
    filename, color_space, rotate_angle, flip_mode, brightness, contrast, gamma,
    filter_type, threshold_type, threshold_value,
    bitwise_op, background_removal_method, change_bg_mode,
    bg_color, bg_image_name_dropdown_value, collage_mode, # bg_image_name_dropdown_value es el valor del dropdown
    detect_contours_flag, detect_faces_flag
):
    global background_mask_global, processed_foreground_global, original_image_for_transparent_export_global

    image = load_image(IMAGE_DIR, filename)
    if image is None:
        # Retorna imágenes negras y una máscara vacía si no se puede cargar la imagen
        black_image = np.zeros((300, 300, 3), dtype=np.uint8)
        empty_mask = np.zeros((300, 300), dtype=np.uint8)
        return black_image, black_image, empty_mask

    original_image = image.copy()
    original_image_for_transparent_export_global = original_image.copy() # Guardar la original para exportación

    current_processed_image = original_image.copy() # Imagen que se va modificando
    background_mask = None # Esta será la máscara del primer plano (objeto=255, fondo=0)

    # --- 1. Eliminación de fondo y obtención de máscara ---
    try:
        if background_removal_method == "HSV":
            processed_foreground_result, background_mask = remove_background_hsv(original_image)
        elif background_removal_method == "LAB":
            processed_foreground_result, background_mask = remove_background_lab(original_image)
        elif background_removal_method == "GrabCut":
            processed_foreground_result, background_mask = grabcut(original_image)
        else: # Si background_removal_method es "None"
            processed_foreground_result = original_image.copy() # No se elimina el fondo
            background_mask = np.zeros(original_image.shape[:2], dtype=np.uint8) # Máscara vacía
        
        # Actualizar globales para la exportación
        background_mask_global = background_mask # Máscara del primer plano (objeto=255)
        processed_foreground_global = processed_foreground_result # Imagen con fondo negro (si se eliminó)
        
    except Exception as e:
        print(f"ERROR en eliminación de fondo ({background_removal_method}): {e}")
        background_mask_global = np.zeros(original_image.shape[:2], dtype=np.uint8)
        processed_foreground_global = original_image.copy() # Fallback a original si falla


    # --- 2. Cambiar fondo si hay máscara ---
    if background_mask_global is not None and np.any(background_mask_global): # Asegurarse de que la máscara no esté vacía
        try:
            if change_bg_mode == "Color" and bg_color:
                bgr_color = hex_to_rgb(bg_color)[::-1] # Convertir RGB a BGR
                current_processed_image = change_background_color(original_image, background_mask_global, bgr_color)
            elif change_bg_mode == "Image" and bg_image_name_dropdown_value: # Usar el valor del dropdown
                bg_image = load_image(BACKGROUND_DIR, bg_image_name_dropdown_value)
                if bg_image is not None:
                    current_processed_image = change_background_image(original_image, background_mask_global, bg_image)
                else:
                    current_processed_image = processed_foreground_global # Fallback a imagen con fondo negro
            else:
                current_processed_image = processed_foreground_global
        except Exception as e:
            print(f"ERROR en cambio de fondo ({change_bg_mode}): {e}")
            current_processed_image = processed_foreground_global # Fallback
    else:
        if background_removal_method == "None":
            current_processed_image = original_image.copy()
        else:
            current_processed_image = processed_foreground_global


    # --- 3. Aplicar transformaciones restantes ---
    if current_processed_image is not None:
        if len(current_processed_image.shape) == 2: # Si es escala de grises
            current_processed_image = cv2.cvtColor(current_processed_image, cv2.COLOR_GRAY2BGR)
        elif len(current_processed_image.shape) == 4: # Si es RGBA (raro aquí, por si acaso)
            current_processed_image = cv2.cvtColor(current_processed_image, cv2.COLOR_BGRA2BGR)

    try:
        if current_processed_image is not None:
            # Operaciones de color
            current_processed_image = convert_color(current_processed_image, color_space)
            if len(current_processed_image.shape) == 2: # Si después de convert_color es GRAYSCALE
                current_processed_image = cv2.cvtColor(current_processed_image, cv2.COLOR_GRAY2BGR)

            # Correcciones
            current_processed_image = rotate(current_processed_image, rotate_angle)
            current_processed_image = flip(current_processed_image, flip_mode)
            current_processed_image = adjust_brightness_contrast(current_processed_image, brightness, contrast)
            current_processed_image = gamma_correction(current_processed_image, gamma)

            # Ecualización de histograma (aplicar solo si es escala de grises o convertir temporalmente)
            if color_space == "GRAYSCALE":
                current_processed_image = equalize_histogram(current_processed_image)
                if len(current_processed_image.shape) == 2:
                    current_processed_image = cv2.cvtColor(current_processed_image, cv2.COLOR_GRAY2BGR)
            
            # Filtros
            if filter_type != "None":
                current_processed_image = apply_filter(current_processed_image, filter_type)
                if len(current_processed_image.shape) == 2:
                    current_processed_image = cv2.cvtColor(current_processed_image, cv2.COLOR_GRAY2BGR)

            # Umbrales (requieren imagen en escala de grises)
            if threshold_type != "None":
                if len(current_processed_image.shape) == 3:
                    gray_for_threshold = cv2.cvtColor(current_processed_image, cv2.COLOR_BGR2GRAY)
                else:
                    gray_for_threshold = current_processed_image.copy()

                if threshold_type == "Binary":
                    current_processed_image = apply_threshold(gray_for_threshold, threshold_value)
                elif threshold_type == "Adaptive":
                    current_processed_image = adaptive_threshold(gray_for_threshold)
                elif threshold_type == "Otsu":
                    current_processed_image = otsu_threshold(gray_for_threshold)
                
                if len(current_processed_image.shape) == 2:
                    current_processed_image = cv2.cvtColor(current_processed_image, cv2.COLOR_GRAY2BGR)

            # Operaciones bitwise (solo NOT implementado de forma simple aquí)
            if bitwise_op == "NOT":
                current_processed_image = bitwise_not(current_processed_image)

            # Detección (contornos y rostros)
            if detect_contours_flag:
                current_processed_image = detect_contours(current_processed_image)
            if detect_faces_flag:
                current_processed_image = detect_faces_haar(current_processed_image)
            
    except Exception as e:
        print(f"ERROR durante las operaciones de procesamiento: {e}")
        current_processed_image = original_image.copy()


    # --- 4. Collage (se aplica al final, puede cambiar el tamaño de la imagen) ---
    # Asegurarse de que las imágenes para el collage estén en RGB
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # current_processed_image está en BGR en este punto, convertir a RGB para el collage
    processed_rgb_for_collage = current_processed_image
    if processed_rgb_for_collage is not None:
        if len(processed_rgb_for_collage.shape) == 2:
            processed_rgb_for_collage = cv2.cvtColor(processed_rgb_for_collage, cv2.COLOR_GRAY2BGR)
        processed_rgb_for_collage = cv2.cvtColor(processed_rgb_for_collage, cv2.COLOR_BGR2RGB)
    else:
        processed_rgb_for_collage = np.zeros((300, 300, 3), dtype=np.uint8) # Imagen negra si es nula

    if collage_mode == "Original vs Procesada (Horizontal)":
        current_processed_image = stack_images([original_image_rgb, processed_rgb_for_collage], cols=2)
    elif collage_mode == "Original vs Procesada (Vertical)":
        current_processed_image = stack_images([original_image_rgb, processed_rgb_for_collage], cols=1)
    elif collage_mode == "Procesada (Horizontal)":
        current_processed_image = stack_images([processed_rgb_for_collage, processed_rgb_for_collage], cols=2)
    elif collage_mode == "Procesada (Vertical)":
        current_processed_image = stack_images([processed_rgb_for_collage, processed_rgb_for_collage], cols=1)


    # Convertir a RGB para mostrar en Gradio y retornar también la máscara para depuración
    display_mask = background_mask_global # La máscara del primer plano
    if display_mask is None:
        display_mask = np.zeros(original_image.shape[:2], dtype=np.uint8) # Máscara vacía si no hay
    if len(display_mask.shape) == 3: # Si por alguna razón la máscara tiene 3 canales
        display_mask = cv2.cvtColor(display_mask, cv2.COLOR_BGR2GRAY)
    
    if current_processed_image is not None:
        if len(current_processed_image.shape) == 3 and current_processed_image.shape[2] == 3:
            return original_image_rgb, cv2.cvtColor(current_processed_image, cv2.COLOR_BGR2RGB), display_mask
        elif len(current_processed_image.shape) == 2:
            return original_image_rgb, cv2.cvtColor(current_processed_image, cv2.COLOR_GRAY2RGB), display_mask
    
    print("ADVERTENCIA: Formato de imagen final inesperado o imagen nula. Devolviendo imagen original y máscara vacía.")
    return original_image_rgb, cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), np.zeros(original_image.shape[:2], dtype=np.uint8)


def export_transparent_object():
    global original_image_for_transparent_export_global, background_mask_global
    
    if original_image_for_transparent_export_global is None or background_mask_global is None:
        print("ERROR: No se ha procesado una imagen con eliminación de fondo o no hay máscara disponible para exportar el objeto transparente.")
        return None

    try:
        bgr_image = original_image_for_transparent_export_global.copy()
        alpha_channel = background_mask_global.copy() # Esta es la máscara del objeto (foreground)

        if alpha_channel.shape[:2] != bgr_image.shape[:2]:
            alpha_channel = cv2.resize(alpha_channel, (bgr_image.shape[1], bgr_image.shape[0]))
            _, alpha_channel = cv2.threshold(alpha_channel, 127, 255, cv2.THRESH_BINARY)

        rgba_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2BGRA)
        rgba_image[:, :, 3] = alpha_channel # El objeto será opaco (255), el fondo transparente (0)

        save_path = "objeto_transparente.png"
        cv2.imwrite(save_path, rgba_image)
        print(f"Objeto transparente guardado en: {save_path}")
        return save_path
    except Exception as e:
        print(f"ERROR al exportar objeto transparente: {e}")
        return None


def main_interface():
    image_list = list_images()
    bg_list = list_backgrounds()

    with gr.Blocks() as demo:
        gr.Markdown("## 🖼️ Aplicación Completa de Procesamiento de Imágenes")

        with gr.Row():
            with gr.Column():
                image_selector = gr.Dropdown(label="Imagen base", choices=image_list, value=image_list[0] if image_list else None)
                color_space = gr.Dropdown(label="Espacio de color", choices=["RGB", "HSV", "LAB", "GRAYSCALE"], value="RGB")

                gr.Markdown("### ⚙️ Controles de Transformación")
                with gr.Accordion("Rotación y Volteo", open=False):
                    rotate_angle = gr.Slider(-180, 180, value=0, label="Rotación")
                    flip_mode = gr.Radio(["horizontal", "vertical", "both"], label="Modo volteo", value="horizontal")
                
                with gr.Accordion("Brillo, Contraste y Gamma", open=False):
                    brightness = gr.Slider(-100, 100, value=0, label="Brillo")
                    contrast = gr.Slider(-100, 100, value=0, label="Contraste")
                    gamma = gr.Slider(0.1, 3.0, value=1.0, step=0.1, label="Gamma")

                with gr.Accordion("Filtros y Umbrales", open=False):
                    filter_type = gr.Dropdown(label="Filtro", choices=["None", "blur", "gaussian", "bilateral", "median", "sharpen", "sobel", "laplacian", "canny", "emboss", "custom"], value="None")
                    threshold_type = gr.Radio(["None", "Binary", "Adaptive", "Otsu"], label="Tipo de umbral", value="None")
                    threshold_value = gr.Slider(0, 255, value=128, label="Valor umbral (binario)")
                    bitwise_op = gr.Radio(["None", "AND", "OR", "NOT"], label="Operación bitwise", value="None")

                with gr.Accordion("Eliminación y Cambio de Fondo", open=True):
                    background_removal_method = gr.Radio(["None", "HSV", "LAB", "GrabCut"], label="Eliminación de fondo", value="None")
                    
                    change_bg_mode = gr.Radio(["None", "Color", "Image"], label="Cambio de fondo", value="None")
                    bg_color = gr.ColorPicker(label="Color fondo")
                    bg_image_name = gr.Dropdown(label="Imagen fondo (cambio)", choices=bg_list, visible=False) # Este es el dropdown que se muestra/oculta

                with gr.Accordion("Collage y Detección", open=True): # Abrir por defecto para que sea visible
                    collage_mode = gr.Radio([
                        "None", 
                        "Original vs Procesada (Horizontal)", 
                        "Original vs Procesada (Vertical)",
                        "Procesada (Horizontal)",
                        "Procesada (Vertical)"
                    ], label="Modo collage", value="None")
                    detect_contours_flag = gr.Checkbox(label="Detectar contornos")
                    detect_faces_flag = gr.Checkbox(label="Detectar rostros (Haar cascades)")

                process_button = gr.Button("Procesar Imagen")
                download_btn = gr.Button("Descargar recorte PNG transparente")


            with gr.Column():
                output_original = gr.Image(label="Imagen Original", type="numpy", height=400) # Re-añadido
                output_image = gr.Image(label="Resultado", type="numpy", height=400)
                mask_display = gr.Image(label="Máscara de Primer Plano (Objeto Blanco, Fondo Negro)", type="numpy", height=200)
                download_file_output = gr.File(label="Objeto Recortado (PNG Transparente)", file_count="single", visible=False)


        inputs = [
            image_selector, color_space, rotate_angle, flip_mode, brightness, contrast, gamma,
            filter_type, threshold_type, threshold_value, bitwise_op, background_removal_method,
            change_bg_mode, bg_color, bg_image_name, collage_mode,
            detect_contours_flag, detect_faces_flag
        ]

        # Lógica para mostrar/ocultar el selector de imagen de fondo
        def toggle_bg_image_selector(change_bg):
            return gr.update(visible=(change_bg == "Image"))

        change_bg_mode.change(fn=toggle_bg_image_selector, inputs=change_bg_mode, outputs=bg_image_name)

        # Conectar el botón de procesar a la función principal
        process_button.click(
            fn=process_all,
            inputs=inputs,
            outputs=[output_original, output_image, mask_display] # Actualizados los outputs
        )

        # Conectar los cambios de los inputs a la función principal para actualización en tiempo real
        for inp in inputs:
            inp.change(
                fn=process_all, # Ahora llama a process_all directamente
                inputs=inputs,
                outputs=[output_original, output_image, mask_display] # Actualizados los outputs
            )

        # Lógica para la descarga del objeto transparente
        download_btn.click(
            fn=export_transparent_object,
            inputs=[],
            outputs=download_file_output
        ).then(
            lambda file_path: gr.update(visible=file_path is not None),
            inputs=download_file_output,
            outputs=download_file_output
        )

    return demo

if __name__ == "__main__":
    if not list_images():
        print(f"Advertencia: No se encontraron imágenes en el directorio '{IMAGE_DIR}'.")
        print("Por favor, añade algunas imágenes (ej. .jpg, .png) a esta carpeta para que la aplicación funcione.")
        print("La aplicación se iniciará, pero no habrá imágenes para seleccionar.")
    if not list_backgrounds():
        print(f"Advertencia: No se encontraron imágenes en el directorio '{BACKGROUND_DIR}'.")
        print("Por favor, añade algunas imágenes de fondo a esta carpeta si planeas usar la función de cambio de fondo por imagen.")

    try:
        demo = main_interface()
        demo.launch()
        print("\n¡Aplicación Gradio lanzada exitosamente!")
        print("Accede a ella en tu navegador usando la URL que aparece arriba (normalmente http://127.0.0.1:7860).")
    except Exception as e:
        print(f"\nERROR CRÍTICO al lanzar la aplicación Gradio: {e}")
        print("Esto podría deberse a un puerto ocupado, problemas de red o una instalación corrupta de Gradio.")
        print("Intenta:")
        print("1. Reinstalar Gradio con 'pip install --upgrade --force-reinstall gradio'")
        print("2. Probar con un puerto diferente: 'python app.py --port 8000'")
        print("3. Borrar la caché de tu navegador o usar una ventana de incógnito.")
