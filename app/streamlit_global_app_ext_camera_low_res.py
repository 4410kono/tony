from pathlib import Path
import streamlit as st
from ultralytics import YOLO
from utils_ext_camera_low_res import play_webcam
from PIL import Image


st.set_page_config(
    page_title="Fall Detector",

    initial_sidebar_state="expanded"
)

# Load the image for the title
title_image = Image.open("camera_icon.png")

# Display the title image
st.image(title_image, use_column_width=True)

# sidebar
st.sidebar.header("Model Config")
model = YOLO('best.pt')

# image/video options
st.sidebar.header("Input Config")
source_selectbox = st.sidebar.selectbox(
    "Select Source",
    ["Webcam", "Image", "Video"]
)

source_img = None
if source_selectbox == "Video": # Video
    pass
elif source_selectbox == "Image": # Image
    pass
elif source_selectbox == "Webcam": # Webcam
    play_webcam(conf=0.5, model=model)
else:
    st.error("Currently only 'Image' and 'Video' source are implemented")
