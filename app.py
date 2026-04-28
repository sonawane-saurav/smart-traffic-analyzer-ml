import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import time

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Traffic Analyzer", layout="wide")
st.title("Smart Traffic Analyzer System")

# -----------------------------
# Load Model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# -----------------------------
# Utility Functions
# -----------------------------
def process_frame(frame, model):
    try:
        results = model(frame)[0]
    except:
        return frame, {'car':0,'truck':0,'bus':0,'motorcycle':0}, 0

    counts = {
        'car': 0,
        'truck': 0,
        'bus': 0,
        'motorcycle': 0
    }

    try:
        for box in results.boxes:
            cls = int(box.cls[0])
            label = results.names[cls]

            if label in counts:
                counts[label] += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    except:
        pass

    total = sum(counts.values())
    return frame, counts, total


def get_density(total):
    if total < 5:
        return "Low"
    elif total < 15:
        return "Medium"
    else:
        return "High"


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Controls")

video_file = st.sidebar.file_uploader("Upload Traffic Video", type=["mp4"])
start = st.sidebar.button("Start Analysis")

# -----------------------------
# Main Logic
# -----------------------------
if video_file is not None and start:

    # Save video
    with open("temp.mp4", "wb") as f:
        f.write(video_file.read())

    cap = cv2.VideoCapture("temp.mp4")

    if not cap.isOpened():
        st.error("Error opening video file")
        st.stop()

    frame_counts = []
    total_vehicles = 0

    video_placeholder = st.empty()
    col1, col2 = st.columns([2, 1])

    start_time = time.time()

    frame_limit = 150
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret or frame_idx >= frame_limit:
            break

        if frame is None:
            continue

        frame_idx += 1

        # Resize (performance boost)
        try:
            frame = cv2.resize(frame, (640, 360))
        except:
            continue

        frame, counts, total = process_frame(frame, model)
        density = get_density(total)

        total_vehicles += total
        frame_counts.append(total)

        # Convert safely
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except:
            continue

        # FINAL SAFETY CHECK
        if frame is None or not isinstance(frame, np.ndarray):
            continue
        if len(frame.shape) != 3:
            continue

        # -----------------------------
        # LEFT: Video
        # -----------------------------
        with col1:
            video_placeholder.image(frame, channels="RGB", use_container_width=True)

        # -----------------------------
        # RIGHT: Stats
        # -----------------------------
        with col2:
            st.markdown("### 📊 Vehicle Stats")
            st.metric("Cars", counts['car'])
            st.metric("Bikes", counts['motorcycle'])
            st.metric("Trucks", counts['truck'])
            st.metric("Buses", counts['bus'])
            st.metric("Total", total)

            st.markdown("### 🚦 Traffic Status")

            if density == "High":
                st.error("🚨 HIGH TRAFFIC")
            elif density == "Medium":
                st.warning("⚠️ MODERATE TRAFFIC")
            else:
                st.success("✅ SMOOTH TRAFFIC")

        # FPS
        fps = 1 / (time.time() - start_time)
        start_time = time.time()
        st.sidebar.metric("FPS", f"{fps:.2f}")

    cap.release()

    # -----------------------------
    # Graph
    # -----------------------------
    st.markdown("## 📈 Traffic Trend Over Time")

    if len(frame_counts) > 0:
        fig, ax = plt.subplots()
        ax.plot(frame_counts)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Vehicle Count")
        ax.set_title("Traffic Flow Analysis")
        st.pyplot(fig)

    st.success(f"Total Vehicles Detected: {total_vehicles}")

else:
    st.info("Upload a video and click Start Analysis")
