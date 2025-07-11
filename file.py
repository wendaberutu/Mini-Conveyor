import os
from pathlib import Path
import shutil

# Direktori asal campuran
source_dir = Path(r"C:\Users\WALETA-PROJECT-PC\Downloads\barang jadi badan\barang jadi badan")  # ganti sesuai folder asalmu

# Folder tujuan
images_dir = Path("dataset/images")
labels_dir = Path("dataset/labels")

# Buat folder jika belum ada
images_dir.mkdir(parents=True, exist_ok=True)
labels_dir.mkdir(parents=True, exist_ok=True)

# Proses semua file
for file in source_dir.iterdir():
    if file.suffix.lower() in [".jpg", ".png"]:
        shutil.move(str(file), str(images_dir / file.name))
        print(f"📁 Pindah gambar: {file.name}")
    elif file.suffix.lower() == ".json":
        shutil.move(str(file), str(labels_dir / file.name))
        print(f"📝 Pindah label: {file.name}")
    else:
        print(f"⚠️ File diabaikan: {file.name}")
