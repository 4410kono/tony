from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import tempfile
from config import *
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, webrtc_streamer, WebRtcMode, RTCConfiguration
import numpy as np
import av
import base64
import time
import os
from slack_sdk import WebhookClient


def _display_detected_frames(conf, model, st_count, st_frame, image):
    """
    Display the detected objects on a video frame using the YOLOv8 model.
    :param conf (float): Confidence threshold for object detection.
    :param model (YOLOv8): An instance of the `YOLOv8` class containing the YOLOv8 model.
    :param st_frame (Streamlit object): A Streamlit object to display the detected video.
    :param image (numpy array): A numpy array representing the video frame.
    :return: None
    """
    # Resize the image to a standard size
    #image = cv2.resize(image, (720, int(720 * (9 / 16))))

    # Predict the objects in the image using YOLOv8 model
    res = model.predict(image, conf=conf)

    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )

def play_webcam(conf, model):   # Streamlit on cloud (global)
    """
    Plays a webcam stream on cloud. Detects Objects in real-time using the YOLO object detection model.

    Returns:
        None

    Raises:
        None
    """
    # st.sidebar.title("Webcam Object Detection")

    st.text("External-camera lower solution")

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")

        orig_h, orig_w = image.shape[0:2]

        print(f"Current Frame Resolution: Width = {orig_w}, Height = {orig_h}")

        width = 720  # Set the desired width for processing

        # cv2.resize used in a forked thread may cause memory leaks
        processed_image = np.asarray(Image.fromarray(image).resize((width, int(width * orig_h / orig_w))))

        if model is not None:
            # Perform object detection using YOLO model
            res = model.predict(processed_image, conf=conf)
            # print(f'resboxes: {res.boxes}')

            # Plot the detected objects on the video frame
            res_plotted = res[0].plot()
            # print(f'resplotted: {res_plotted}')


        return av.VideoFrame.from_ndarray(res_plotted, format="bgr24"), res

    st.write(video_frame_callback)
    print(video_frame_callback)

    # webrtc_streamer(
    #     key="example",
    #     # video_transformer_factory=lambda: MyVideoTransformer(conf, model),
    #     video_frame_callback = video_frame_callback[0],
    #     rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    #     media_stream_constraints={
    #                                 "video": {
    #                                     "width": {"max": 360},
    #                                     "height": {"max": 270}
    #                                 },
    #                                 "audio": False
    #                             },
    # )

    # resultholder = st.empty()

    # client = WebhookClient(os.environ["SLACK_WEBHOOK_URL"])
    # result_object = video_frame_callback[1]
    # # get the class id
    # class_ids = result_object.boxes.cls

    # # get a dictionay of all class names
    # class_names_dict = result_object.names

    # for class_id in class_ids:
    #     class_name = class_names_dict[int(class_id)]
    #     # results.append(class_name)
    #     resultholder.write(class_name)
