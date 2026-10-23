import cv2
import numpy as np
from typing import List, Tuple
import streamlit as st
from ultralytics import YOLO


@st.cache_resource
def load_yolo_model(model_name: str):
    """Caches the YOLO model in memory so it doesn't reload on every frame."""
    return YOLO(model_name)


def pad_and_crop(image: np.ndarray, x1: int, y1: int, x2: int, y2: int, pad_ratio: float = 0.15) -> np.ndarray:
    """
    Adds a 15% contextual border around the detected object so Gemini
    sees surrounding edges, shadows, and surface details.
    """
    h_img, w_img, _ = image.shape
    box_w = x2 - x1
    box_h = y2 - y1

    pad_x = int(box_w * pad_ratio)
    pad_y = int(box_h * pad_ratio)

    nx1 = max(0, x1 - pad_x)
    ny1 = max(0, y1 - pad_y)
    nx2 = min(w_img, x2 + pad_x)
    ny2 = min(h_img, y2 + pad_y)

    return image[ny1:ny2, nx1:nx2]


def detect_objects_yolo(
    image: np.ndarray,
    model_name: str = "yolo26s.pt",
    conf_threshold: float = 0.35,
    iou_threshold: float = 0.45,
    custom_classes: List[str] | None = None,
) -> Tuple[np.ndarray, List[str], List[np.ndarray]]:
    """
    Runs high-accuracy YOLO26 / YOLO-World detection with contextual crop padding.
    """
    model = load_yolo_model(model_name)

    # If using YOLO-World, set user's custom open-vocabulary prompts
    if "world" in model_name and custom_classes:
        model.set_classes(custom_classes)

    results = model.predict(
        image,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=640,
        verbose=False
    )[0]

    annotated = image.copy()
    labels = []
    crops = []

    # Sort detections by highest confidence first
    boxes = sorted(results.boxes, key=lambda b: float(b.conf[0]), reverse=True)

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        class_name = results.names[cls_id]

        crop = pad_and_crop(image, x1, y1, x2, y2, pad_ratio=0.15)
        if crop.size == 0:
            continue

        label_str = f"#{idx+1} {class_name} ({int(conf*100)}%)"
        labels.append(label_str)
        crops.append(crop)

        # Draw high-visibility bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            label_str,
            (x1, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )

    return annotated, labels, crops


def extract_regions_of_interest(
    image: np.ndarray,
    min_area: int = 2000,
    max_area: int = 500000
) -> Tuple[np.ndarray, List[str], List[np.ndarray]]:
    """
    Enhanced OpenCV pipeline using CLAHE contrast normalization + Canny edges.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply CLAHE to boost local contrast in dim indoor lighting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 140)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    labels = []
    crops = []
    annotated = image.copy()

    roi_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            roi_count += 1
            x, y, w, h = cv2.boundingRect(cnt)
            label_str = f"ROI #{roi_count} ({int(area)} px)"
            labels.append(label_str)
            crops.append(pad_and_crop(image, x, y, x + w, y + h, pad_ratio=0.10))

            cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 200, 0), 2)
            cv2.putText(
                annotated, label_str, (x, max(20, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2
            )

    return annotated, labels, crops