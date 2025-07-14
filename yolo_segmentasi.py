from ultralytics import YOLO
import numpy as np

class YoloSegmentor:
    def __init__(self, model_path, confidence):
        self.model = YOLO(model_path)
        self.classList = ["person"]  # Ubah sesuai kelas kamu
        self.confidence = confidence

    def segment(self, image):
        results = self.model.predict(image, conf=self.confidence, iou=0.5, task="segment")
        result = results[0]
        return self.extract_segments(result)

    def extract_segments(self, result):
        segments = []

        # Pastikan model output punya mask
        if not hasattr(result, "masks") or result.masks is None:
            return segments

        masks = result.masks.data.cpu().numpy()  # [N, H, W]
        boxes = result.boxes
        class_names = result.names

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if class_names[cls_id] not in self.classList:
                continue

            mask = masks[i]  # 2D mask untuk objek ke-i
            segments.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "class_id": cls_id,
                "confidence": conf,
                "mask": mask
            })

        return segments
