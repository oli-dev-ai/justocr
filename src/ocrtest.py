import pytesseract
import pipeline

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'



def testocr(image_path):
    image = pipeline.process_pipeline(image_path)
    text = pytesseract.image_to_string(image, lang="eng+pol")
    print(text)
    return text

if __name__=="__main__":
    testocr("../data/samples/samplemarei.jpeg")


 


