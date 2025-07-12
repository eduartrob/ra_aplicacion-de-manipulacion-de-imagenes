import cv2
import numpy as np

def change_background_color(foreground_img, background_mask, color=(0, 255, 0)):
    """
    Cambia el fondo de la imagen por un color sólido.
    - foreground_img: imagen original (BGR)
    - background_mask: máscara binaria donde fondo=0, objeto=255
    - color: tuple BGR del color de fondo deseado
    """
    bg_color_img = np.full(foreground_img.shape, color, dtype=np.uint8)
    bg_mask = cv2.bitwise_not(background_mask)
    fg = cv2.bitwise_and(foreground_img, foreground_img, mask=background_mask)
    bg = cv2.bitwise_and(bg_color_img, bg_color_img, mask=bg_mask)
    return cv2.add(fg, bg)

def change_background_image(foreground_img, background_mask, new_background):
    """
    Cambia el fondo de la imagen por otra imagen.
    - foreground_img: imagen original (BGR)
    - background_mask: máscara binaria donde fondo=0, objeto=255
    - new_background: imagen de fondo (BGR) para reemplazar (debe ser mismo tamaño)
    """
    if new_background.shape[:2] != foreground_img.shape[:2]:
        new_background = cv2.resize(new_background, (foreground_img.shape[1], foreground_img.shape[0]))
    bg_mask = cv2.bitwise_not(background_mask)
    fg = cv2.bitwise_and(foreground_img, foreground_img, mask=background_mask)
    bg = cv2.bitwise_and(new_background, new_background, mask=bg_mask)
    return cv2.add(fg, bg)

def change_background(foreground_img, background_mask, new_background=None, color=None):
    """
    Cambia el fondo usando un color sólido o una imagen.
    - new_background: imagen BGR para usar como fondo nuevo (opcional)
    - color: tuple BGR para usar como fondo de color (opcional)
    
    Al menos uno debe ser especificado.
    """
    if new_background is not None:
        return change_background_image(foreground_img, background_mask, new_background)
    elif color is not None:
        return change_background_color(foreground_img, background_mask, color)
    else:
        raise ValueError("Debes especificar un color o una imagen para el fondo nuevo.")
