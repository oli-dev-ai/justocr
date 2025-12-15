import cv2
import matplotlib.pyplot as plt
import numpy as np

def load_image(image_path):
    """Ładuje obraz z dysku."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")
    return image

def convert_to_grayscale(image):
    """Konwertuje do grayscale."""
    if image is None:
        raise ValueError("Image cannot be None")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def display_image(image, title="Image", grayscale=False):
    """Wyświetla obraz."""
    if grayscale:
        plt.imshow(image, cmap='gray')
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
    
    plt.title(title)
    plt.axis('off')
    plt.show()

def denoise(image):
    """Removes noise from image"""
    if image is None:
        raise ValueError("Image cannot be none")
    denoised = cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)
    return denoised

def threshold(image):
    """Adds threshold to image"""
    if image is None:
        raise ValueError("Image cannot be none")
    if len(image.shape) != 2:
        raise ValueError(f"Expected grayscale image (2D), got: {image.shape}")
    binary = cv2.adaptiveThreshold(
        image,       #input grayscale
        255,         #Max Value (białe piksele)
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, #Metoda - Gaussian weighted sum
        cv2.THRESH_BINARY, # Typ: binary threshold
        11,         #Block size(musi być nieparzyste)
        2           #Constant odejmowania od średniej
    )
    return binary

def correct_rotation(image):
    """Image rotatoin"""
    if image is None:
        raise ValueError("Image cannot be None")
    
    coords = np.column_stack(np.where(image > 0))  # wykrywanie konturów
    
    if len(coords) < 5:   # Za mało punktów, zwróć oryginalny
        return image
    
    angle = cv2.minAreaRect(coords)[-1] # Dopasuj minimalny prostokąt
    
    if angle < -45: # Normalizuj kąt do -45 do 45 stopni
        angle = 90 + angle
    
    if abs(angle) < 0.5: # Jeśli kąt jest bardzo mały, nie rotuj
        return image
    
    (h, w) = image.shape[:2]  # Rotacja
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), 
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    
    return rotated