import cv2
import numpy as np

def stack_images(images, cols=2, size=(300, 300)):
    if not images:
        raise ValueError("La lista de imágenes está vacía")

    # Redimensionar todas las imágenes
    resized_images = [cv2.resize(img, size) for img in images]

    # Calcular filas necesarias
    rows = []
    for i in range(0, len(resized_images), cols):
        row_imgs = resized_images[i:i+cols]
        # Si faltan imágenes para completar la fila, agregar imágenes negras
        if len(row_imgs) < cols:
            blank_img = np.zeros_like(row_imgs[0])
            row_imgs += [blank_img] * (cols - len(row_imgs))
        row = np.hstack(row_imgs)
        rows.append(row)

    # Apilar filas verticalmente
    collage = np.vstack(rows)
    return collage
