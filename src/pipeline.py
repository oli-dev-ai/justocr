from opencv import load_image, convert_to_grayscale, display_image, denoise, threshold, correct_rotation
import cv2
from pathlib import Path

def process_pipeline(image_path):
    img = load_image(image_path)
    gray = convert_to_grayscale(img)
    denoised = denoise(gray)
    thresholded = threshold(denoised)
    corrected_rot = correct_rotation(thresholded)

    output_dir = Path("../data/processed")

    input_path = Path(image_path)
    base_name = input_path.stem     #wyciągniecie nazwy pliku bez rozszerzenia
    extension = input_path.suffix   #wyciągnięcie tylko rozszerzenia pliku

    counter = 1
    while True:
        output_filename = f"{base_name}_processed_{counter}{extension}"
        output_path = output_dir / output_filename   # Operator / łączy ścieżki 
        
        if not output_path.exists():                 #sprawdza czy plik nie istnieje, jesli true to zostawia obecny numer
            break
        
        counter += 1
    
    cv2.imwrite(str(output_path), corrected_rot)
    print(f"Saved in{output_path}")

    return corrected_rot