import cv2
import matplotlib.pyplot as plt

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