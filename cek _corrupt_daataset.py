from PIL import Image
import os

image_dir = r"D:\wenda\Mini Conveyor\dataset\images"

def fix_jpeg_images(folder):
    fixed = 0
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg')):
                path = os.path.join(root, file)
                try:
                    img = Image.open(path)
                    img.save(path, "JPEG", quality=95)
                    fixed += 1
                except Exception as e:
                    print(f"[SKIPPED] {file} -> {e}")
    print(f"✅ Selesai: {fixed} file diperbaiki ulang.")

fix_jpeg_images(image_dir)
