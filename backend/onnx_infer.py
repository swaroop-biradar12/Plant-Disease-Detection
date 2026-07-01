"""
Lightweight YOLO ONNX inference — no torch, no ultralytics.
Uses onnxruntime + numpy + opencv only.

Handles two possible export formats automatically:
  A) End2End / NMS-embedded export: output shape (1, N, 6)
     each row = [x1, y1, x2, y2, confidence, class_id]  (already NMS'd)
  B) Raw export: output shape (1, 4+num_classes, num_boxes)
     needs manual box decode + NMS
"""

import numpy as np
import cv2
import onnxruntime as ort


class ONNXDetector:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        self.input_size = (
            input_shape[2] if isinstance(input_shape[2], int) else 640,
            input_shape[3] if isinstance(input_shape[3], int) else 640,
        )

        self.names = {}
        meta = self.session.get_modelmeta().custom_metadata_map
        if "names" in meta:
            try:
                names_raw = eval(meta["names"], {"__builtins__": {}})
                self.names = {int(k): v for k, v in names_raw.items()}
            except Exception:
                self.names = {}

        print(f"[ONNXDetector] input_size={self.input_size}, names={self.names}")

    def _letterbox(self, image, new_shape):
        h, w = image.shape[:2]
        new_h, new_w = new_shape
        r = min(new_h / h, new_w / w)
        resized_w, resized_h = int(round(w * r)), int(round(h * r))
        resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

        pad_w = new_w - resized_w
        pad_h = new_h - resized_h
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        return padded, r, left, top

    def predict(self, pil_image, conf_threshold: float = 0.5, iou_threshold: float = 0.45):
        image = np.array(pil_image.convert("RGB"))
        orig_h, orig_w = image.shape[:2]

        padded, ratio, pad_x, pad_y = self._letterbox(image, self.input_size)

        blob = padded.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[None, ...]

        raw = self.session.run(None, {self.input_name: blob})[0]
        preds = raw[0]  # drop batch dim

        # ── Detect output format ─────────────────────────────────────────────
        # Format A: End2End, shape (N, 6) -> [x1,y1,x2,y2,score,class_id], already NMS'd
        if preds.ndim == 2 and preds.shape[1] == 6:
            return self._parse_end2end(preds, conf_threshold, ratio, pad_x, pad_y)

        # Format B: raw, shape (4+num_classes, num_boxes) or (num_boxes, 4+num_classes)
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T  # -> (num_boxes, 4+num_classes)

        return self._parse_raw(preds, conf_threshold, iou_threshold, ratio, pad_x, pad_y)

    def _parse_end2end(self, preds, conf_threshold, ratio, pad_x, pad_y):
        results = []
        for row in preds:
            x1, y1, x2, y2, score, cls_id = row[:6]
            if score < conf_threshold:
                continue
            cls_id = int(cls_id)
            results.append({
                "disease": self.names.get(cls_id, f"class_{cls_id}"),
                "confidence": round(float(score), 3),
            })
        return results

    def _parse_raw(self, preds, conf_threshold, iou_threshold, ratio, pad_x, pad_y):
        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]

        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]

        # If scores look like raw logits (outside 0-1), apply sigmoid
        if confidences.max() > 1.0 or confidences.min() < 0.0:
            confidences = 1 / (1 + np.exp(-confidences))

        mask = confidences >= conf_threshold
        boxes_xywh = boxes_xywh[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        results = []
        if len(boxes_xywh) == 0:
            return results

        xc, yc, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
        x1 = (xc - w / 2 - pad_x) / ratio
        y1 = (yc - h / 2 - pad_y) / ratio
        x2 = (xc + w / 2 - pad_x) / ratio
        y2 = (yc + h / 2 - pad_y) / ratio

        nms_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        indices = cv2.dnn.NMSBoxes(nms_boxes, confidences.tolist(), conf_threshold, iou_threshold)

        if len(indices) == 0:
            return results

        indices = np.array(indices).flatten()
        for i in indices:
            cls_id = int(class_ids[i])
            results.append({
                "disease": self.names.get(cls_id, f"class_{cls_id}"),
                "confidence": round(float(confidences[i]), 3),
            })

        return results