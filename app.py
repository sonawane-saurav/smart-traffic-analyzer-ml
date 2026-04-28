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
# Load Model (safe)
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# -----------------------------
# Processing Function
# -----------------------------
def process_frame(frame):
    results = model(frame)[0]

    counts = {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0}

    for box in results.boxes:
        cls = int(box.cls[0])
        label = results.names[cls]

        if label in counts:
            counts[label] += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
# Main
# -----------------------------
if video_file and start:

    # Save file
    with open("temp.mp4", "wb") as f:
        f.write(video_file.read())

    cap = cv2.VideoCapture("temp.mp4")

    if not cap.isOpened():
        st.error("Cannot open video")
        st.stop()

    video_placeholder = st.empty()
    col1, col2 = st.columns([2, 1])

    frame_counts = []
    total_vehicles = 0

    frame_limit = 120
    frame_idx = 0

    start_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret or frame_idx >= frame_limit:
            break

        frame_idx += 1

        # Resize safely
        frame = cv2.resize(frame, (640, 360))

        # Process
        frame, counts, total = process_frame(frame)

        total_vehicles += total
        frame_counts.append(total)

        # Convert BGR → RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # GUARANTEED VALID DISPLAY
        if isinstance(frame, np.ndarray) and frame.ndim == 3:

            with col1:
                video_placeholder.image(frame, channels="RGB", use_container_width=True)

            with col2:
                st.markdown("### 📊 Vehicle Stats")
                st.metric("Cars", counts['car'])
                st.metric("Bikes", counts['motorcycle'])
                st.metric("Trucks", counts['truck'])
                st.metric("Buses", counts['bus'])
                st.metric("Total", total)

                density = get_density(total)

                st.markdown("### 🚦 Traffic Status")
                if density == "High":
                    st.error("🚨 HIGH TRAFFIC")
                elif density == "Medium":
                    st.warning("MODERATE TRAFFIC")
                else:
                    st.success("SMOOTH TRAFFIC")

        # FPS
        fps = 1 / (time.time() - start_time)
        start_time = time.time()
        st.sidebar.metric("FPS", f"{fps:.2f}")

    cap.release()

    # -----------------------------
    # Graph
    # -----------------------------
    if frame_counts:
        st.markdown("## 📈 Traffic Trend")

        fig, ax = plt.subplots()
        ax.plot(frame_counts)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Vehicle Count")
        st.pyplot(fig)

    st.success(f"Total Vehicles Detected: {total_vehicles}")

else:
    st.info("Upload a video and click Start Analysis")
