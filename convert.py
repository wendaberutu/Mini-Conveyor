import os
import json
import shutil
import random
from pathlib import Path
from PIL import Image

# === KONFIGURASI ===
json_dir = Path("dataset/labels")
image_dir = Path("dataset/images")

# Persentase partisi
val_ratio = 0.2
test_ratio = 0.1
train_ratio = 1.0 - val_ratio - test_ratio

# Buat folder output
for split in ["train", "val", "test"]:
    (json_dir / split).mkdir(parents=True, exist_ok=True)
    (image_dir / split).mkdir(parents=True, exist_ok=True)

# Ambil dan acak semua file json
json_files = list(json_dir.glob("*.json"))
random.shuffle(json_files)

# Hitung batas index
total = len(json_files)
n_train = int(total * train_ratio)
n_val = int(total * val_ratio)

splits = {
    "train": json_files[:n_train],
    "val": json_files[n_train:n_train + n_val],
    "test": json_files[n_train + n_val:]
}

# === FUNGSI KONVERSI ===
def convert_json_to_txt(json_path, image_path, output_txt_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    if not data.get("shapes"):
        print(f"⚠️  Tidak ada anotasi dalam: {json_path.name}")
        return

    img_w, img_h = data["imageWidth"], data["imageHeight"]
    yolo_lines = []

    for shape in data["shapes"]:
        if shape["shape_type"] != "polygon":
            continue
        class_id = 0  # Ganti jika multi-kelas
        points = shape["points"]
        norm = []
        for x, y in points:
            nx, ny = round(x / img_w, 6), round(y / img_h, 6)
            norm.extend([nx, ny])
        line = f"{class_id} " + " ".join(map(str, norm))
        yolo_lines.append(line)

    with open(output_txt_path, "w") as f:
        f.write("\n".join(yolo_lines))

# === PROSES PARTISI DAN KONVERSI ===
for split, files in splits.items():
    print(f"\n📁 Memproses {split}: {len(files)} file")
    for json_file in files:
        # Tentukan jalur gambar
        with open(json_file, "r") as f:
            data = json.load(f)
        image_filename = data["imagePath"]
        image_path = image_dir / image_filename
        if not image_path.exists():
            print(f"⚠️  Gambar tidak ditemukan: {image_filename}")
            continue

        # Copy file
        shutil.copy2(json_file, json_dir / split / json_file.name)
        shutil.copy2(image_path, image_dir / split / image_filename)

        # Konversi json ke txt
        txt_name = json_file.stem + ".txt"
        output_txt = json_dir / split / txt_name
        convert_json_to_txt(json_file, image_path, output_txt)

        print(f"✅ {json_file.name} → {txt_name}")

print("\n✅ Semua selesai.")
