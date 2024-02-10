import pytesseract
from PIL import Image
from img_process import process_image



# Path to the Tesseract-OCR executable (change this according to your installation)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_from_image(image_path):
    # Open the image file
    with Image.open(image_path) as img:
        # Use pytesseract to extract text from the image
        extracted_text = pytesseract.image_to_string(img)
    return extracted_text


if __name__ == "__main__":
    # Provide the path to your image
    image_path = r"C:\Users\glitcher\Desktop\teseract_ocr\processed_imgs\processed_image.jpg"
    extracted_text = extract_text_from_image(image_path)
    
    print("Extracted Text:")
    print(extracted_text)
