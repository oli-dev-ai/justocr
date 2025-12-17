import pytesseract
import pipeline
from pathlib import Path

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def save_ocr(text, processed_filename):
    """
    Zapisuje OCR z taką samą nazwą jak processed image.

    Args:
        text (str): Wyekstraktowany tekst
        original_image_path (str): Ścieżka do oryginalnego obrazu
    """
    output_dir = Path("../data/txt_reco")                       #gdzie ma się zapisać
    
    txt_filename = Path(processed_filename).stem + ".txt"       #nazwa -> nazwa bazowa + txt_proc
    txt_path = output_dir / txt_filename                        #gdzie ma się zapisać i jako co

    with open(txt_path,'w', encoding='utf-8') as f:
        f.write(text)
    
    print(f"Saved OCR result: {txt_path}")

def testocr(image_path):
    """OCR z automatycznym zapisem img preprocessed oraz txt"""
    processed_image, processed_filename = pipeline.process_pipeline(image_path)  #zwraca nazwe pliku oraz obraz
    text = pytesseract.image_to_string(processed_image, lang="pol")              #zwraca ocr
    save_ocr(text, processed_filename)                                           #uzycie funkcji zapisania do txt

    return text

if __name__=="__main__":
    testocr("../data/samples/samplemarei.jpeg")


 


