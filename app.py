import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

from ultralytics import YOLO
from utils import process_frame, get_density

model = YOLO("yolov8n.pt", task='detect')

st.set_page_config(page_title="Traffic Analyzer", layout="wide")
st.title("Smart Traffic Analyzer System")

st.sidebar.header("Controls")
video_file = st.sidebar.file_uploader("Upload Traffic Video", type=["mp4"])
start = st.sidebar.button("Start Analysis")

if video_file is not None and start:

    # Save uploaded video
    with open("temp.mp4", "wb") as f:
        f.write(video_file.read())

    cap = cv2.VideoCapture("temp.mp4")

    # Data storage for graph
    frame_counts = []

    # Layout
    col1, col2 = st.columns([2, 1])

    video_placeholder = col1.empty()
    stats_placeholder = col2.empty()
    density_placeholder = col2.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, counts, total = process_frame(frame, model)
        density = get_density(counts)

        frame_counts.append(total)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        video_placeholder.image(frame, channels="RGB", use_container_width=True)

        with stats_placeholder.container():
            st.markdown("### Vehicle Stats")
            st.metric("Cars", counts['car'])
            st.metric("Bikes", counts['motorcycle'])
            st.metric("Trucks", counts['truck'])
            st.metric("Buses", counts['bus'])
            st.metric("Total", total)

        if density == "Low":
            density_placeholder.success("Traffic: LOW")
        elif density == "Medium":
            density_placeholder.warning("Traffic: MEDIUM")
        else:
            density_placeholder.error("Traffic: HIGH")

    cap.release()
    st.markdown("## Traffic Trend Over Time")

    if len(frame_counts) > 0:
        fig, ax = plt.subplots()

        ax.plot(frame_counts, color='blue', linewidth=2)

        ax.set_xlabel("Frame")
        ax.set_ylabel("Vehicle Count")
        ax.set_title("Traffic Flow Analysis")

        ax.grid(True)

        st.pyplot(fig)
    else:
        st.warning("No data available for graph")

else:
    st.info("Upload a video and click Start Analysis")