from ultralytics import YOLO
import numpy as np
import cv2
import torch
import torch.hub
import warnings

warnings.filterwarnings("ignore")

# --- YOLOv8 Segmentasi ---
class YoloSegmentor:
    def __init__(self, model_path, confidence=0.3):
        self.model = YOLO(model_path)
        self.classList = ["sarang"]
        self.confidence = confidence

    def segment(self, image):
        results = self.model.predict(image, conf=self.confidence, iou=0.5, task="segment", verbose=False)
        result = results[0]
        return self.extract_segments(result)

    def extract_segments(self, result):
        segments = []
        if not hasattr(result, "masks") or result.masks is None:
            return segments
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes
        class_names = result.names

        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if class_names[cls_id] not in self.classList:
                continue
            mask = masks[i]
            segments.append({
                "class_id": cls_id,
                "confidence": conf,
                "mask": mask
            })
        return segments


# --- MiDaS Depth Estimator ---
class DepthEstimator:
    def __init__(self):
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
        self.model.eval()

    def estimate(self, image):
        input_batch = self.transform(image)
        with torch.no_grad():
            prediction = self.model(input_batch)
            depth = prediction.squeeze().cpu().numpy()
            
        depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return depth, depth_vis

    def estimate_height_from_mask(self, depth_map, mask):
        # Resize mask agar sama dengan ukuran depth_map
        mask_resized = cv2.resize(mask.astype(np.uint8), (depth_map.shape[1], depth_map.shape[0]), interpolation=cv2.INTER_NEAREST)
        depth_values = depth_map[mask_resized > 0]
        if len(depth_values) == 0:
            return None
        min_depth = float(np.min(depth_values))
        max_depth = float(np.max(depth_values))
        height = max_depth - min_depth
        return round(height, 4)


# --- MAIN ---
def main():
    # Inisialisasi model
    detector = YoloSegmentor(
        model_path=r"D:\wenda\Mini Conveyor\runs\segment\model_segmentation\weights\best.pt",
        confidence=0.3
    )
    depth_estimator = DepthEstimator()

    # Buka kamera
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("[ERROR] Tidak bisa membuka kamera.")
        return

    print("[INFO] Menangkap satu gambar dan menghitung tinggi sarang...")

    # Tangkap satu gambar
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("[WARNING] Gagal membaca frame dari kamera.")
        return

    resized_frame = cv2.resize(frame, (640, 480))
    segmen = detector.segment(resized_frame)
    depth_map, depth_vis = depth_estimator.estimate(resized_frame)

    overlay = resized_frame.copy()
    for i, segment in enumerate(segmen):
        mask = segment["mask"]
        mask = cv2.resize(mask.astype(np.uint8), (resized_frame.shape[1], resized_frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Visualisasi mask
        mask_colored = np.zeros_like(resized_frame)
        mask_colored[mask > 0] = [0, 255, 0]
        overlay = cv2.addWeighted(overlay, 1.0, mask_colored, 0.5, 0)

        # Estimasi tinggi dari depth map
        height = depth_estimator.estimate_height_from_mask(depth_map, mask)
        if height is not None:
            cv2.putText(
                overlay, f"Tinggi: {height}", (10, 30 + 30 * i),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
            print(f"[INFO] Tinggi sarang ke-{i+1}: {height}")

    # Gabung hasil segmentasi dan depth untuk ditampilkan
    depth_vis_resized = cv2.resize(depth_vis, (resized_frame.shape[1], resized_frame.shape[0]))
    depth_vis_resized = cv2.cvtColor(depth_vis_resized, cv2.COLOR_GRAY2BGR)
    stacked = np.hstack([overlay, depth_vis_resized])

    cv2.imshow("Hasil Segmentasi & Estimasi Tinggi", stacked)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
