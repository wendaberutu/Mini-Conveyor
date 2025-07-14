from ultralytics import YOLO
import numpy as np
import cv2
import torch


class YoloSegmentor:
    def __init__(self, model_path, confidence):
        self.model = YOLO(model_path)
        self.classList = ["sarang"]  
        self.confidence = confidence

    def segment(self, image):
        results = self.model.predict(image, conf=self.confidence, iou=0.5, task="segment")
        result = results[0]
        return self.extract_segments(result)
    
    def extract_segments(self, result):
        segments = []

        if not hasattr(result, "masks") or result.masks is None:
            print("Tidak ada mask yang terdeteksi.")
            return segments
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes
        class_names = result.names

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # Debug: tampilkan nama kelas
            print(f"Deteksi kelas: {class_names[cls_id]}, confidence: {conf:.2f}")

            # Jika ingin semua kelas tampil, bisa hapus filter ini
            if class_names[cls_id] not in self.classList:
                continue
            mask = masks[i]
            segments.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "class_id": cls_id,
                "confidence": conf,
                "mask": mask    
            })
        return segments
#class SizeKaki :




def main():
    detector = YoloSegmentor(
        model_path=r"D:\wenda\Mini Conveyor\runs\segment\model_segmentation\weights\best.pt",
        confidence=0.3
    )   

    image = cv2.imread(r"D:\wenda\Mini Conveyor\dataset\images\test\sarang_gbj-1747638385334.jpg")
    if image is None:
        print("Gambar tidak ditemukan. Cek path.")
        return

    print("Image loaded:", image.shape)

    segmen = detector.segment(image)
    print("Jumlah segmentasi:", len(segmen))
    if len(segmen) == 0:
        print("Tidak ada objek terdeteksi.")

    overlay = image.copy()
    for segment in segmen:
        bbox = segment["bbox"]
        mask = segment["mask"]
        class_id = segment["class_id"]
        confidence = segment["confidence"]

        print("BBox:", bbox)
        print("Mask unique:", np.unique(mask))

        # Resize mask ke ukuran gambar asli
        mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_colored = np.zeros_like(image)
        mask_colored[mask > 0] = [0, 255, 0]
        overlay = cv2.addWeighted(overlay, 1.0, mask_colored, 0.5, 0)

        cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(overlay, f"Class: {class_id}, Conf: {confidence:.2f}", 
                    (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("Segmented Image", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()