import os
from PIL import Image
import shutil

def classify_and_count_images(source_folder, prefix_0, prefix_1):
    count_total = 0
    count_prefix_0 = 0
    count_prefix_1 = 0
    first_image_dimensions = None

    # Directories for categorized images
    dir_prefix_0 = os.path.join(source_folder, f'label_{prefix_0}')
    dir_prefix_1 = os.path.join(source_folder, f'label_{prefix_1}')

    # Create directories if they don't exist
    if not os.path.exists(dir_prefix_0):
        os.makedirs(dir_prefix_0)
    if not os.path.exists(dir_prefix_1):
        os.makedirs(dir_prefix_1)

    for filename in os.listdir(source_folder):
        if filename.endswith('.png'):
            count_total += 1
            file_path = os.path.join(source_folder, filename)

            # Get dimensions of the first image
            if count_total == 1:
                with Image.open(file_path) as img:
                    first_image_dimensions = img.size

            # Classify and copy the image
            if filename.startswith(f'{prefix_0}_'):
                count_prefix_0 += 1
                shutil.copy(file_path, dir_prefix_0)
            elif filename.startswith(f'{prefix_1}_'):
                count_prefix_1 += 1
                shutil.copy(file_path, dir_prefix_1)

    return count_total, count_prefix_0, count_prefix_1, first_image_dimensions

def main():
    source_folder = 'labelImage'  # Adjust the folder path as needed
    prefix_0 = '0'  # Prefix for the first category
    prefix_1 = '1'  # Prefix for the second category

    total, count_0, count_1, first_dimensions = classify_and_count_images(source_folder, prefix_0, prefix_1)

    print(f'Total images: {total}')
    print(f'Images with {prefix_0}_ prefix: {count_0}')
    print(f'Images with {prefix_1}_ prefix: {count_1}')
    print(f'Dimensions of the first image (width x height): {first_dimensions[0]} x {first_dimensions[1]}')

if __name__ == "__main__":
    main()