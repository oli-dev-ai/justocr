from opencv import load_image, convert_to_grayscale, display_image

import cv2
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Załaduj RAZ
    img = load_image("../data/samples/samplejop.jpg")
    
    # Test 1: Wyświetl oryginalny
    display_image(img, title="Original")
    
    # Test 2: Konwertuj i wyświetl grayscale
    gray = convert_to_grayscale(img)
    display_image(gray, title="Grayscale", grayscale=True)
    
    # Test 3: Możesz dalej używać tego samego obrazu
    # denoised = denoise(img)
    # display_image(denoised, title="Denoised")
