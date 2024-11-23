import os
from PIL import Image, ImageFilter
import cv2
import numpy as np


# Folder wejściowy
input_folder = 'HR'

# Lista metod
methods = [
    'Bilinear',
    'Bicubic',
    'Lanczos_a3',
    'Lanczos_a4',
    'Nearest_Neighbour',
    'Nearest_Neighbour_Gaussian',
    'Bilinear_Gaussian'
]


for method in methods:
    output_folder = method
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


image_files = [
    f for f in os.listdir(input_folder)
    if os.path.isfile(os.path.join(input_folder, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
]


scale_factor = 0.25 

for image_file in image_files:
    
    image_path = os.path.join(input_folder, image_file)
    image_pil = Image.open(image_path)
    width, height = image_pil.size
    new_size = (int(width * scale_factor), int(height * scale_factor))


    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    # Metoda Bilinear
    if 'Bilinear' in methods:
        resized_image = cv2.resize(image_cv, new_size, interpolation=cv2.INTER_LINEAR)
        output_path = os.path.join('Bilinear', image_file)
        cv2.imwrite(output_path, resized_image)

    # Metoda Bicubic
    if 'Bicubic' in methods:
        resized_image = cv2.resize(image_cv, new_size, interpolation=cv2.INTER_CUBIC)
        output_path = os.path.join('Bicubic', image_file)
        cv2.imwrite(output_path, resized_image)

    # Metoda Lanczos a=3 (PIL)
    if 'Lanczos_a3' in methods:
        resized_image_pil = image_pil.resize(new_size, resample=Image.LANCZOS)
        output_path = os.path.join('Lanczos_a3', image_file)
        resized_image_pil.save(output_path)

    # Metoda Lanczos a=4 (OpenCV)
    if 'Lanczos_a4' in methods:
        resized_image = cv2.resize(image_cv, new_size, interpolation=cv2.INTER_LANCZOS4)
        output_path = os.path.join('Lanczos_a4', image_file)
        cv2.imwrite(output_path, resized_image)

    # Metoda Nearest Neighbour
    if 'Nearest_Neighbour' in methods:
        resized_image = cv2.resize(image_cv, new_size, interpolation=cv2.INTER_NEAREST)
        output_path = os.path.join('Nearest_Neighbour', image_file)
        cv2.imwrite(output_path, resized_image)

    # Nearest Neighbour z filtrem Gaussa
    if 'Nearest_Neighbour_Gaussian' in methods:
        blurred_image = cv2.GaussianBlur(image_cv, (5, 5), 0.5)
        resized_image = cv2.resize(blurred_image, new_size, interpolation=cv2.INTER_NEAREST)
        output_path = os.path.join('Nearest_Neighbour_Gaussian', image_file)
        cv2.imwrite(output_path, resized_image)

    # Bilinear z filtrem Gaussa
    if 'Bilinear_Gaussian' in methods:
        blurred_image = cv2.GaussianBlur(image_cv, (5, 5), 0.5)
        resized_image = cv2.resize(blurred_image, new_size, interpolation=cv2.INTER_LINEAR)
        output_path = os.path.join('Bilinear_Gaussian', image_file)
        cv2.imwrite(output_path, resized_image)

