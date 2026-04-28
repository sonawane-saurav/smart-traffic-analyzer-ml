import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Traffic Analyzer", layout="wide")
st.title("Smart Traffic Analyzer System")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

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

    # Save uploaded video
    with open("temp.mp4", "wb") as f:
        f.write(video_file.read())

    cap = cv2.VideoCapture("temp.mp4")

    if not cap.isOpened():
        st.error("❌ Unable to open video")
        st.stop()

    frame_counts = []
    total_vehicles = 0

    frame_limit = 120
    frame_idx = 0

    first_frame = None

    # -----------------------------
    # PROCESS VIDEO (SAFE)
    # -----------------------------
    while True:
        ret, frame = cap.read()

        if not ret or frame_idx >= frame_limit:
            break

        frame_idx += 1

        if frame is None:
            continue

        # Resize for performance
        frame = cv2.resize(frame, (640, 360))

        # Save ONE processed frame with bounding boxes
        if first_frame is None:
            temp_frame = frame.copy()

            try:
                results = model(temp_frame)[0]

                for box in results.boxes:
                    label = results.names[int(box.cls[0])]

                    if label in ['car', 'truck', 'bus', 'motorcycle']:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(temp_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(temp_frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                first_frame = temp_frame

            except:
                first_frame = frame.copy()

        # YOLO detection (for counting only)
        try:
            results = model(frame)[0]
        except:
            continue

        counts = {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0}

        try:
            for box in results.boxes:
                label = results.names[int(box.cls[0])]
                if label in counts:
                    counts[label] += 1
        except:
            continue

        total = sum(counts.values())
        total_vehicles += total
        frame_counts.append(total)

    cap.release()

    # -----------------------------
    # DISPLAY
    # -----------------------------
    col1, col2 = st.columns([2, 1])

    # Show processed frame (with boxes)
    if first_frame is not None:
        try:
            first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
            with col1:
                st.image(first_frame, caption="Detected Vehicles", use_container_width=True)
        except:
            pass

    # Stats
    with col2:
        st.markdown("### 📊 Summary")
        st.metric("Total Vehicles", total_vehicles)
        st.metric("Frames Processed", len(frame_counts))

        if len(frame_counts) > 0:
            avg = int(np.mean(frame_counts))
            st.metric("Avg Vehicles / Frame", avg)

            if avg < 5:
                st.success("✅ Low Traffic")
            elif avg < 15:
                st.warning("⚠️ Medium Traffic")
            else:
                st.error("🚨 High Traffic")

    # -----------------------------
    # Graph
    # -----------------------------
    if len(frame_counts) > 0:
        st.markdown("## 📈 Traffic Trend")

        fig, ax = plt.subplots()
        ax.plot(frame_counts)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Vehicle Count")
        ax.set_title("Traffic Flow Analysis")

        st.pyplot(fig)

else:
    st.info("Upload a video and click Start Analysis")
