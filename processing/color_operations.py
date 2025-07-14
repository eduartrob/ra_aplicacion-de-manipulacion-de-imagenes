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