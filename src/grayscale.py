import cv2

def convert_to_grayscale(image):
    """
    Converts a color image to grayscale
    
    Args:
        image (numpy.ndarray): BGR image from OpenCV
    
    Returns:
        numpy.ndarray: Grayscale image, or None if an error occurs
    """
    if image is None:
         raise ValueError("Image cannot be none")
    if len(image.shape) != 3:
        raise ValueError(f"Image needs to be in color (3 canals), received: {image.shape}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    return gray