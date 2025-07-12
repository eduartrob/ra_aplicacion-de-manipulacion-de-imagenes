import cv2
import numpy as np

def rotate(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

def flip(image, mode='horizontal'):
    """
    mode: 'horizontal', 'vertical', 'both'
    """
    if mode == 'horizontal':
        flipCode = 1
    elif mode == 'vertical':
        flipCode = 0
    elif mode == 'both':
        flipCode = -1
    else:
        raise ValueError("Modo no válido. Usa 'horizontal', 'vertical' o 'both'")
    return cv2.flip(image, flipCode)

def resize(image, width=None, height=None):
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    elif height is None:
        r = width / float(w)
        dim = (width, int(h * r))
    else:
        dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    """
    brightness: [-255,255]
    contrast: [-127,127]
    """
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        image = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127*(1 - f)
        image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)

    return image

def gamma_correction(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def equalize_histogram(image, is_color=True):
    if is_color:
        channels = cv2.split(image)
        eq_channels = [cv2.equalizeHist(ch) for ch in channels]
        return cv2.merge(eq_channels)
    else:
        return cv2.equalizeHist(image)
