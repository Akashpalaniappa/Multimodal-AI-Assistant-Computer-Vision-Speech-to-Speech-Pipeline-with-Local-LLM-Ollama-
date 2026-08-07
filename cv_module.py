"""
Computer Vision module using YOLOv8 (Ultralytics) for real-time object detection.
"""

from ultralytics import YOLO
from config import YOLO_MODEL_PATH, YOLO_CONFIDENCE

_model = None


def load_cv_model():
    """Load YOLOv8 model once and cache it."""
    global _model
    if _model is None:
        print(f"[CV] Loading YOLO model: {YOLO_MODEL_PATH}")
        _model = YOLO(YOLO_MODEL_PATH)
    return _model


def detect_objects(image_path: str) -> list[dict]:
    """
    Run object detection on an image and return structured results.

    Args:
        image_path: path to the image file on disk

    Returns:
        list of dicts: [{"label": "laptop", "confidence": 0.91}, ...]
    """
    model = load_cv_model()
    results = model.predict(image_path, conf=YOLO_CONFIDENCE, verbose=False)

    detections = []
    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            detections.append({"label": label, "confidence": round(confidence, 2)})

    print(f"[CV] Detected: {detections}")
    return detections


def detections_to_context(detections: list[dict]) -> str:
    """
    Convert raw detections into a short natural-language string
    the LLM can use as context. E.g. "laptop, coffee cup, notebook"
    """
    if not detections:
        return "no recognizable objects"
    # de-duplicate while keeping order
    labels = list(dict.fromkeys(d["label"] for d in detections))
    return ", ".join(labels)


if __name__ == "__main__":
    # quick manual test: python modules/cv_module.py path/to/image.jpg
    import sys
    if len(sys.argv) > 1:
        dets = detect_objects(sys.argv[1])
        print(detections_to_context(dets))
    else:
        print("Usage: python cv_module.py <image_file_path>")
