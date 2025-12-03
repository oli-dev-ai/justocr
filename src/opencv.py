import cv2
import matplotlib.pyplot as plt

def load_display(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Loading {image_path} was not succesful :(")
        return
    print("Image was loaded")
    print(f"Image size: {image.shape}")
    print(f"Image type: {image.dtype}")
    #Function below is converting from BGR to RGB because matplotlib uses RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.imshow(image_rgb)
    plt.title("Oryginalny obraz")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # ten if powoduje że jeśli ten plik jest ładowany bezpośrednio to automatycznie się wykonuje
    # natomiast jeśli bedzie importowany w innym pliku to nie wykona sie automatycznie
    load_display("../data/samples/crm-wzor1.jpg")
