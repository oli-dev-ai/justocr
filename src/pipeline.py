from opencv import load_image, convert_to_grayscale, display_image, denoise, threshold, correct_rotation

def process_pipeline(image_path):
    img = load_image(image_path)
    gray = convert_to_grayscale(img)
    denoised = denoise(gray)
    thresholded = threshold(denoised)
    corrected_rot = correct_rotation(thresholded)
    return corrected_rot