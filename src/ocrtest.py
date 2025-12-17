import pytesseract
import pipeline
from pathlib import Path

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def save_ocr(text, original_image_path):
    """
    Zapisuje wynik OCR do txt.
    
    Args:
        text (str): Wyekstraktowany tekst
        original_image_path (str): Ścieżka do oryginalnego obrazu
    """
    output_dir = Path("../data/txt_reco")             #gdzie ma się zapisać
    original_image_path = Path(original_image_path)   #oryginala scieżka pliku do ocr
    base_name = original_image_path.stem              #nazwa bazowa -> original img
    txt_filename = f"{base_name}_txt_proc.txt"        #nazwa -> nazwa bazowa + txt_proc
    txt_path = output_dir / txt_filename              #gdzie ma się zapisać i jako co

    with open(txt_path,'w', encoding='utf-8') as f:
        f.write(text)

def testocr(image_path):
    """OCR z automatycznym zapisem img preprocessed oraz txt"""
    image = pipeline.process_pipeline(image_path)
    text = pytesseract.image_to_string(image, lang="pol")
    save_ocr(text, image_path)  #uzycie funkcji zapisania do txt

    return text

if __name__=="__main__":
    testocr("../data/samples/samplemarei.jpeg")


 


