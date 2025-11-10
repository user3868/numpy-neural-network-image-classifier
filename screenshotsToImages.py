import os
from PIL import Image


def split_images(source_folder, dest_folder, grid_width, grid_height):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for filename in os.listdir(source_folder):
        if filename.endswith('.png'):
            with Image.open(os.path.join(source_folder, filename)) as img:
                img_width, img_height = img.size

                x_splits = img_width
                y_splits = img_height
                for x in range(x_splits):
                    for y in range(y_splits):
                        left = x * grid_width
                        top = y * grid_height
                        right = left + grid_width
                        bottom = top + grid_height

                        cropped_img = img.crop((left, top, right, bottom))
                        cropped_img_name = f'({left},{top})_{filename}'
                        cropped_img.save(os.path.join(dest_folder, cropped_img_name))


def resize_images(source_folder, dest_folder, new_width, new_height):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for filename in os.listdir(source_folder):
        if filename.endswith('.png'):
            with Image.open(os.path.join(source_folder, filename)) as img:
                resize_img = img.resize((new_width, new_height))
                resize_img.save(os.path.join(dest_folder, filename))


def main():
    source_folder = 'Screenshots'
    grid_width = 100
    grid_height = 100
    new_width = 50
    new_height = 50

    split_dest_folder = f'subImage_{grid_width}_{grid_height}'
    resize_dest_folder = f'sizeImage_{grid_width}_{grid_height}'

    split_images(source_folder, split_dest_folder, grid_width, grid_height)
    resize_images(split_dest_folder, resize_dest_folder, new_width, new_height)


if __name__ == "__main__":
    main()
