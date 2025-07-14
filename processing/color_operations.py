import cv2

def convert_color(image, color_space):
    if color_space == "RGB":
        return image  # No conversión, ya está en BGR
    elif color_space == "HSV":
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif color_space == "LAB":
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    elif color_space == "GRAYSCALE":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        return image


def split_channels(image):
    return cv2.split(image)

def merge_channels(ch1, ch2, ch3):
    return cv2.merge([ch1, ch2, ch3])

def hex_to_rgb(hex_color):
    if not hex_color or not isinstance(hex_color, str) or not hex_color.startswith("#") or len(hex_color) != 7:
        return (0, 0, 0)  # Color por defecto (negro)

    hex_color = hex_color.lstrip("#")
    try:
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    except ValueError:
        return (0, 0, 0)

