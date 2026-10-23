import streamlit as st
import cv2
import numpy as np
from src.vision_core import detect_objects_yolo, extract_regions_of_interest
from src.analyzer import VisionInspectorAgent

st.set_page_config(page_title="Multimodal Vision Inspector", layout="wide")
st.title("🔍 High-Precision YOLO26 + Gemini Vision Inspector")
st.caption("HD Real-Time Object Detection (YOLO26 / Open-Vocab YOLO-World) + Contextual Gemini VLM Diagnostics")

if "captured_crops" not in st.session_state:
    st.session_state.captured_crops = []
if "captured_labels" not in st.session_state:
    st.session_state.captured_labels = []

with st.sidebar:
    st.header("🧠 Detection Engine")
    engine = st.radio(
        "Select Vision Engine:",
        [
            "🚀 YOLO26 (Standard 80 Classes)",
            "🌍 YOLO-World (Custom Text Prompts)",
            "📐 OpenCV CLAHE Contour Extractor"
        ]
    )

    custom_classes = None
    if engine == "🚀 YOLO26 (Standard 80 Classes)":
        model_name = st.selectbox(
            "Model Accuracy Tier:",
            ["yolo26s.pt", "yolo26m.pt", "yolo26n.pt", "yolo11s.pt"],
            index=0,
            help="yolo26s (Small) and yolo26m (Medium) are much more accurate than Nano (n)."
        )
        conf_thresh = st.slider("Confidence Threshold", 0.15, 0.90, 0.35, step=0.05)

    elif engine == "🌍 YOLO-World (Custom Text Prompts)":
        model_name = "yolov8s-worldv2.pt"
        custom_prompt_str = st.text_input(
            "Type ANY objects to detect (comma-separated):",
            value="person, glasses, watch, pen, circuit board, phone, headphone, water bottle"
        )
        custom_classes = [c.strip() for c in custom_prompt_str.split(",") if c.strip()]
        conf_thresh = st.slider("Confidence Threshold", 0.10, 0.90, 0.25, step=0.05)

    else:
        min_area = st.slider("Min Contour Area (px)", 500, 50000, 2500, step=500)

    st.divider()
    st.header("⚙️ Input Source")
    mode = st.radio(
        "Select Input Mode:",
        ["🎥 Live Real-Time Stream (720p HD)", "📸 Browser Camera Snapshot", "📁 Upload Image File"]
    )

    max_rois = st.slider("Max Objects to Send to Gemini", 1, 5, 3)
    context_prompt = st.text_area(
        "Gemini Inspection Context",
        value="Inspect this cropped region in detail. Identify the exact item, brand/type if visible, physical condition, and any defects."
    )


def process_frame(frame: np.ndarray):
    if engine == "🚀 YOLO26 (Standard 80 Classes)":
        return detect_objects_yolo(frame, model_name=model_name, conf_threshold=conf_thresh)
    elif engine == "🌍 YOLO-World (Custom Text Prompts)":
        return detect_objects_yolo(frame, model_name=model_name, conf_threshold=conf_thresh, custom_classes=custom_classes)
    else:
        return extract_regions_of_interest(frame, min_area=min_area)


def run_ai_diagnostics(crops, labels, context_prompt, max_rois):
    st.divider()
    st.subheader("3. Multimodal Gemini Diagnostic Reports")
    inspector = VisionInspectorAgent()

    for idx, (crop, label) in enumerate(zip(crops[:max_rois], labels[:max_rois])):
        with st.container(border=True):
            c_img, c_rep = st.columns([1, 3])
            with c_img:
                st.markdown(f"**{label}**")
                st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), width="stretch")
            with c_rep:
                with st.spinner(f"Analyzing {label} with Gemini..."):
                    enriched_context = f"Detector identified this region as '{label}'. {context_prompt}"
                    report, used_model = inspector.analyze_crop(crop, enriched_context)
                    status_icon = (
                        "🟢" if report.overall_status == "PASS"
                        else "🟡" if report.overall_status == "REVIEW_REQUIRED"
                        else "🔴"
                    )
                    st.markdown(f"**Status:** {status_icon} `{report.overall_status}` &nbsp;|&nbsp; **Model:** `{used_model}`")
                    st.write(report.summary)
                    for item in report.findings:
                        st.markdown(
                            f"- **{item.label}** (`{item.severity.upper()}` - {int(item.confidence*100)}% conf): "
                            f"{item.observation}  \n  *Suggested Action:* {item.suggested_action}"
                        )


if mode == "🎥 Live Real-Time Stream (720p HD)":
    col_ctrl1, col_ctrl2 = st.columns([1, 1])
    with col_ctrl1:
        run_stream = st.toggle("▶️ Start HD Live Camera Feed", value=False)
    with col_ctrl2:
        capture_trigger = st.button("⚡ Capture & Lock Current Objects", type="primary", disabled=not run_stream)

    live_placeholder = st.empty()

    if run_stream:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            st.error("Could not open webcam.")
        else:
            while run_stream:
                ret, frame = cap.read()
                if not ret:
                    break

                annotated_frame, labels, crops = process_frame(frame)

                cv2.putText(
                    annotated_frame,
                    f"HD LIVE | Detected: {len(crops)} objects",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

                live_placeholder.image(
                    cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    width="stretch"
                )

                st.session_state.captured_crops = crops
                st.session_state.captured_labels = labels

                if capture_trigger:
                    break

            cap.release()

    if st.session_state.captured_crops:
        detected_list = ", ".join(st.session_state.captured_labels[:max_rois])
        st.success(f"**Locked {len(st.session_state.captured_crops)} objects:** {detected_list}")
        if st.button("🚀 Run Gemini Diagnostics on Locked Objects", type="primary"):
            run_ai_diagnostics(
                st.session_state.captured_crops,
                st.session_state.captured_labels,
                context_prompt,
                max_rois
            )

elif mode == "📸 Browser Camera Snapshot":
    camera_photo = st.camera_input("Take a snapshot")
    if camera_photo:
        file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        annotated_img, labels, crops = process_frame(image)
        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), width="stretch")
        if crops and st.button("🚀 Run Multimodal AI Inspection", type="primary"):
            run_ai_diagnostics(crops, labels, context_prompt, max_rois)

elif mode == "📁 Upload Image File":
    uploaded_file = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        annotated_img, labels, crops = process_frame(image)
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original", width="stretch")
        with col2:
            st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption=f"Detected ({len(crops)})", width="stretch")
        if crops and st.button("🚀 Run Multimodal AI Inspection", type="primary"):
            run_ai_diagnostics(crops, labels, context_prompt, max_rois)