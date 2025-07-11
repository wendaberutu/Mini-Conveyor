from ultralytics import YOLO
import cv2
import numpy as np


# Load gambar
img = cv2.imread(r"D:\wenda\Mini Conveyor\dataset\images\train\img1695280767.jpg")
img_show = img.copy()

# Load model
model = YOLO("best.pt")

# Inference
results = model(img, conf=0.1)

if results[0].masks is not None:
    masks = results[0].masks.data.cpu().numpy()
    print(f"✅ Jumlah mask: {len(masks)}")

    for i, mask in enumerate(masks):
        mask_uint8 = (mask * 255).astype(np.uint8)

        # Resize mask ke ukuran gambar
        mask_resized = cv2.resize(mask_uint8, (img.shape[1], img.shape[0]))

        # Warna acak
        color = np.random.randint(0, 255, (1, 3), dtype=np.uint8).tolist()[0]
        color_mask = np.stack([mask_resized]*3, axis=-1)
        color_mask = (color_mask * np.array(color).reshape(1, 1, 3) / 255).astype(np.uint8)

        # Overlay
        img_show = cv2.addWeighted(img_show, 1.0, color_mask, 0.5, 0)

    # Tampilkan & simpan
    cv2.imshow("Segmented Result", img_show)
    cv2.imwrite("output_segmented.jpg", img_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("❌ Tidak ada mask ditemukan.")
