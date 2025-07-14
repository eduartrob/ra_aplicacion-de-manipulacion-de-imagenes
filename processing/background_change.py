import cv2
import numpy as np

def change_background_color(image, foreground_mask, bg_color):
    """
    Cambia el fondo a un color sólido.
    - image: imagen BGR original.
    - foreground_mask: máscara binaria donde el objeto es 255 y el fondo es 0.
    - bg_color: tupla BGR, ejemplo (0,0,255) rojo.
    """
    if image is None or foreground_mask is None or bg_color is None:
        return image

    # Asegurarse de que la máscara sea binaria (0 o 255) y de un solo canal
    if len(foreground_mask.shape) == 3:
        foreground_mask = cv2.cvtColor(foreground_mask, cv2.COLOR_BGR2GRAY)
    _, foreground_mask = cv2.threshold(foreground_mask, 127, 255, cv2.THRESH_BINARY)

    # Obtener la máscara del fondo (donde el fondo es 255 y el objeto es 0)
    background_mask = cv2.bitwise_not(foreground_mask)

    # Crear una imagen con el nuevo color de fondo del mismo tamaño que la imagen original
    new_bg_image = np.full(image.shape, bg_color, dtype=np.uint8)

    # Extraer el objeto de la imagen original usando la máscara del primer plano
    object_extracted = cv2.bitwise_and(image, image, mask=foreground_mask)

    # Extraer la parte del nuevo fondo que estará en el área del fondo original
    new_background_area = cv2.bitwise_and(new_bg_image, new_bg_image, mask=background_mask)

    # Combinar el objeto extraído con el nuevo fondo
    result = cv2.add(object_extracted, new_background_area)
    return result


def change_background_image(image, foreground_mask, new_background):
    """
    Cambia el fondo a una imagen.
    - image: imagen original BGR.
    - foreground_mask: máscara binaria donde el objeto es 255 y el fondo es 0.
    - new_background: imagen BGR para el nuevo fondo.
    """
    if image is None or foreground_mask is None or new_background is None:
        return image

    # Redimensionar el nuevo fondo para que coincida con la imagen original
    if new_background.shape[:2] != image.shape[:2]:
        new_background = cv2.resize(new_background, (image.shape[1], image.shape[0]))

    # Asegurarse de que la máscara sea binaria (0 o 255) y de un solo canal
    if len(foreground_mask.shape) == 3:
        foreground_mask = cv2.cvtColor(foreground_mask, cv2.COLOR_BGR2GRAY)
    _, foreground_mask = cv2.threshold(foreground_mask, 127, 255, cv2.THRESH_BINARY)

    # Obtener la máscara del fondo (donde el fondo es 255 y el objeto es 0)
    background_mask = cv2.bitwise_not(foreground_mask)

    # Extraer el objeto de la imagen original usando la máscara del primer plano
    object_extracted = cv2.bitwise_and(image, image, mask=foreground_mask)

    # Extraer la parte de la imagen de fondo que estará en el área del fondo original
    new_background_area = cv2.bitwise_and(new_background, new_background, mask=background_mask)

    # Combinar el objeto extraído con el nuevo fondo
    result = cv2.add(object_extracted, new_background_area)
    return result
