from opencv import load_image, convert_to_grayscale, display_image, denoise

import cv2
import matplotlib.pyplot as plt

if __name__ == "__main__":
    img = load_image("../data/samples/samplejop.jpg")
    gray = convert_to_grayscale(img)
    denoised = denoise(gray)
    display_image(denoised)