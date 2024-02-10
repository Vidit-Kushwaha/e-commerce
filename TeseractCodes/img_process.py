import cv2
import numpy as np
import os

def process_image(input_image_path, output_directory, output_filename):
    # Read the image
    image = cv2.imread(input_image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Thresholding
    _, threshold_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert the threshold image
    inverted_image = cv2.bitwise_not(threshold_image)

    # Dilate the inverted image to enhance text regions
    kernel = np.ones((5,5),np.uint8)
    dilated_image = cv2.dilate(inverted_image, kernel, iterations=1)

    # Save processed image
    output_path = os.path.join(output_directory, output_filename)
    cv2.imwrite(output_path, inverted_image)

    return output_path

# Example usage:
input_image_path = r'C:\Users\glitcher\Desktop\teseract_ocr\imgs\captured_frame.jpg'
output_directory = r'C:\Users\glitcher\Desktop\teseract_ocr\processed_imgs'
output_filename = 'processed_image.jpg'

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

processed_image_path = process_image(input_image_path, output_directory, output_filename)
print("Processed image saved at:", processed_image_path)
