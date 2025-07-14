import os
import json
import shutil
import random
from pathlib import Path
from PIL import Image

# === KONFIGURASI ===
# Folder input
json_dir = Path(r"D:\wenda\Mini Conveyor\Hasil_kaki full sarang")         # JSON LabelMe
image_dir = Path(r"D:\wenda\Mini Conveyor\Hasil_kaki full sarang")        # Gambar asli

# Folder output split
json_out_dir = Path("dataset/labels_split")   # JSON + txt
image_out_dir = Path("dataset/images_split")  # Gambar

# Persentase partisi
val_ratio = 0.2
test_ratio = 0.1
train_ratio = 1.0 - val_ratio - test_ratio

# Buat folder output (train, val, test)
for split in ["train", "val", "test"]:
    (json_out_dir / split).mkdir(parents=True, exist_ok=True)
    (image_out_dir / split).mkdir(parents=True, exist_ok=True)

# Ambil dan acak semua file JSON
json_files = list(json_dir.glob("*.json"))
random.shuffle(json_files)

# Hitung jumlah file per partisi
total = len(json_files)
n_train = int(total * train_ratio)
n_val = int(total * val_ratio)

splits = {
    "train": json_files[:n_train],
    "val": json_files[n_train:n_train + n_val],
    "test": json_files[n_train + n_val:]
}

# === FUNGSI KONVERSI JSON -> YOLO TXT ===
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

# === PROSES PARTISI & KONVERSI ===
for split, files in splits.items():
    print(f"\n📁 Memproses {split}: {len(files)} file")
    for json_file in files:
        with open(json_file, "r") as f:
            data = json.load(f)

        image_filename = data["imagePath"]
        image_path = image_dir / image_filename

        if not image_path.exists():
            print(f"⚠️  Gambar tidak ditemukan: {image_filename}")
            continue

        # Salin file JSON dan Gambar
        shutil.copy2(json_file, json_out_dir / split / json_file.name)
        shutil.copy2(image_path, image_out_dir / split / image_filename)

        # Konversi ke .txt YOLO format
        txt_name = json_file.stem + ".txt"
        output_txt = json_out_dir / split / txt_name
        convert_json_to_txt(json_file, image_path, output_txt)

        print(f"✅ {json_file.name} → {txt_name}")

print("\n✅ Semua proses selesai dengan sukses.")
