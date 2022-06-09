import streamlit as st
from PIL import Image
import numpy as np

from NST import NeuralStyleTransfer

img_file_buffer = st.camera_input("Take a picture")
NST = NeuralStyleTransfer()



style_options = np.array([style for style in NST.style_options.values()])

if img_file_buffer is not None:
    # To read image file buffer as a PIL Image:
    style_selection = st.selectbox(label="Select the desired painting style!",
                                   options=style_options)