# 用照片训练，train_video.py,要设置照片路径
# 用视频训练，train_images.py,要设置视频路径
# 查看训练结果，SIBR_viewer.py,要设置outputs路径

import os
from PIL import Image

def convert_png_to_jpg(png_folder_path):
    """
    Convert all PNG images in a given folder to JPEG format.

    Parameters:
    png_folder_path (str): Path to the folder containing PNG images.

    Returns:
    None
    """

    # Ensure the provided path is a valid directory
    if not os.path.isdir(png_folder_path):
        raise ValueError(f"Invalid directory path: {png_folder_path}")

    # Iterate over all files in the specified folder
    for filename in os.listdir(png_folder_path):
        if filename.endswith(".png"):
            png_file_path = os.path.join(png_folder_path, filename)

            # Open the PNG image using PIL
            with Image.open(png_file_path) as img:
                # Define the output JPEG file path by replacing the '.png' extension with '.jpg'
                jpg_file_path = png_file_path.replace(".png", ".jpg")

                # Save the image as JPEG with a quality of 90% (adjust as needed)
                img.save(jpg_file_path, "JPEG", quality=90)

                print(f"Converted '{filename}' to '{jpg_file_path}'")

    print("Conversion completed.")

# Example usage:
convert_png_to_jpg(r"D:\ZhangZhilong\Download\3dgslaptting\data\kongzi\images_2")
