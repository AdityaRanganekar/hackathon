import streamlit as st
import cv2
import torch
import numpy as np
import time
import tempfile
import pygame
import smtplib
import matplotlib.pyplot as plt
import seaborn as sns
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.plots import Annotator, colors

# Load YOLOv5 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DetectMultiBackend("yolov5s.pt", device=device)
model.eval()

# Streamlit UI Setup
st.title("🔍 AI-Powered Crowd Monitoring System")
st.sidebar.header("📁 Upload Video or Use Webcam")
use_webcam = st.sidebar.checkbox("Use Live Webcam 📹")
uploaded_file = st.sidebar.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])
fps_display = st.sidebar.slider("FPS Display Speed", 1, 30, 15)

# Placeholders for video, risk, and graphs
video_placeholder = st.empty()
risk_placeholder = st.empty()
graph_placeholder = st.empty()
heatmap_placeholder = st.empty()

# Data storage for graphs
density_data, movement_data, heatmap_data = [], [], []

# Alert Sound Function
pygame.mixer.init()
try:
    sound = pygame.mixer.Sound("alert_sound.wav")
except pygame.error:
    st.sidebar.warning("⚠️ Alert sound file not found! Please add 'alert_sound.wav'.")

# Twilio Credentials (Replace with actual credentials from your Twilio Console)
TWILIO_SID = "AC5154e6eb043c25afa80facaf3b28c94b"
TWILIO_AUTH_TOKEN = "862a5042cb20e2c5815e6d19a8ed148c"
TWILIO_PHONE_NUMBER = "+18573672071"
USER_PHONE_NUMBER = "+919529211812"  # Your personal number (must be verified in Twilio trial)

# Function to Send SMS Alert
def send_sms_alert():
    try:
        client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
        
        message = client.messages.create(
            body="⚠️ High-Risk Crowd Alert: Immediate action required!",
            from_=TWILIO_PHONE_NUMBER,
            to=USER_PHONE_NUMBER
        )
        
        st.sidebar.success(f"📲 SMS Alert Sent! Message SID: {message.sid}")
    
    except TwilioRestException as e:
        st.sidebar.error(f"❌ Twilio Error: {e}")

# Manual SMS Alert Button
if st.sidebar.button("🚨 Send SMS Alert"):
    send_sms_alert()

# Process Video or Webcam
def process_video(video_source):
    cap = cv2.VideoCapture(video_source)
    prev_gray = None
    alert_played = False
    frame_count = 0
    start_time = time.time()  # Track FPS calculation

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        height, width = frame.shape[:2]
        resized_frame = cv2.resize(frame, (640, 640))

        img = torch.from_numpy(resized_frame).permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0).to(device)

        # YOLO Object Detection
        with torch.no_grad():
            pred = model(img)
        pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.5)

        annotator = Annotator(frame, line_width=2)
        person_count = 0

        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in det:
                    if int(cls) == 0:  # Detect only "person"
                        annotator.box_label(xyxy, f"Person {conf:.2f}", color=colors(int(cls), True))
                        person_count += 1

        # Compute Crowd Density
        crowd_density = person_count / (height * width / 1e6)
        density_data.append(crowd_density)

        # Compute Movement Speed
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        movement_score = 0

        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            movement_score = np.mean(mag)
            movement_data.append(movement_score)

        prev_gray = gray

        # Determine Risk Level
        risk_level = "🟢 LOW"
        if crowd_density > 5.5:
            risk_level = "🟡 MODERATE"
            if movement_score > 2.0:
                risk_level = "🔴 HIGH"
                if not alert_played and "sound" in locals():
                    sound.play()
                    send_sms_alert()
                    alert_played = True
            else:
                alert_played = False  # Reset alert if risk reduces

        # Display FPS
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        # Display Risk Level
        risk_placeholder.markdown(f"""
        ## 🚦 **Risk Level: {risk_level}**
        - 👥 **Crowd Density:** {crowd_density:.2f} people/m²  
        - 🏃 **Movement Speed:** {movement_score:.2f}
        - 🎥 **FPS:** {fps:.2f}
        """, unsafe_allow_html=True)

        # Convert Frame to RGB for Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame, channels="RGB", use_container_width=True)

        # Generate Real-Time Graph & Heatmap
        plot_graph()
        plot_heatmap(movement_score)

        time.sleep(1 / fps_display)

    cap.release()

# Function to Generate Real-Time Graph
def plot_graph():
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(density_data, label="Crowd Density", color="blue")
    ax.plot(movement_data, label="Movement Intensity", linestyle="dashed", color="red")
    ax.set_title("📊 Crowd Risk Analysis")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Value")
    ax.legend()
    graph_placeholder.pyplot(fig)

# Function to Generate Heatmap
def plot_heatmap(movement_score):
    heatmap_data.append(movement_score)
    heatmap_matrix = np.expand_dims(heatmap_data, axis=0)

    fig, ax = plt.subplots(figsize=(6, 2))
    sns.heatmap(heatmap_matrix, cmap="coolwarm", cbar=False, xticklabels=False, yticklabels=False)
    ax.set_title("🔥 Movement Heatmap")
    heatmap_placeholder.pyplot(fig)

# Start Processing
if use_webcam:
    st.warning("📹 **Using Live Webcam... Press 'Stop' to Exit**")
    process_video(0)  # Webcam source
elif uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name
    process_video(temp_path)


