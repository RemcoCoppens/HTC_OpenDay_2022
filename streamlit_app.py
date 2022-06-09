import streamlit as st

import os
import math
import time
import numpy as np
from datetime import date

import matplotlib.pyplot as plt
from matplotlib.offsetbox import (OffsetImage,AnchoredOffsetbox)
import PIL.Image

import tensorflow as tf
import tensorflow_hub as hub

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

class NeuralStyleTransfer():
    def __init__(self, from_camera=False):
        self.content_img_size = (384, 384)
        self.style_img_size = (256, 256)

        if from_camera:
            self.content_path = 'C:/Users/remco/OneDrive/Pictures/Camera Roll'
        else:
            self.content_path = './Content'

        self.processed_content = set(os.listdir(self.content_path))
        self.img_cntr = len(self.processed_content)

        self.style_path = '.\Styles'
        self.style_options = {i: styletype for i, styletype in enumerate(os.listdir(self.style_path))}

        self.hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
        
        self.save_location = './SaveArt'
        self.save_cntr = 0

        print(f'-- Setup Completed --')

    def watermark(self, ax, fig, size_divided=6):
        img = PIL.Image.open('Logo_Datacation.png')
        width, height = ax.figure.get_size_inches() * fig.dpi
        wm_width = int(width/size_divided) # make the watermark 1/4 of the figure size
        scaling = (wm_width / float(img.size[0]))
        wm_height = int(float(img.size[1])*float(scaling))
        img = img.resize((wm_width, wm_height), PIL.Image.ANTIALIAS)

        imagebox = OffsetImage(img, zoom=1, alpha=1.0)
        imagebox.image.axes = ax

        ao = AnchoredOffsetbox(4, pad=0.01, borderpad=0, child=imagebox)
        ao.patch.set_alpha(0)
        ax.add_artist(ao)

    def image_to_tensor(self, path_to_img, img_size):
        """ Load the image and transform it to a Tensor """
        img = tf.io.read_file(path_to_img)
        img = tf.image.decode_image(img, channels=3, dtype=tf.float32)
    
        # Resize the image to specific dimensions
        img = tf.image.resize(img, img_size, preserve_aspect_ratio=True)
        img = img[tf.newaxis, :]
        return img
    
    def tensor_to_image(self, tensor, plot_img=True):
        """ Convert the created tensor to an image and show the image """
        tensor = tensor*255
        tensor = np.array(tensor, dtype=np.uint8)
        tensor = tensor[0]
        if plot_img:
            plt.figure(figsize=(20,10))
            plt.axis('off')
            return plt.imshow(tensor)
        else:
            return tensor
    
    def transform_image(self, content_name):
        """ Retrieve the content and transform it using the chosen style type """
        style_type = self.style_options[int(input(self.style_options))]
        
        content_image_tensor = self.image_to_tensor(path_to_img=f"{self.content_path}\{content_name}", 
                                                    img_size=self.content_img_size)
        style_image_tensor = self.image_to_tensor(path_to_img=f"{self.style_path}\{style_type}",
                                                  img_size=self.style_img_size)

        combined_result = self.hub_module(tf.constant(content_image_tensor), 
                                          tf.constant(style_image_tensor))[0]
        self.tensor_to_image(combined_result)

img_file_buffer = st.camera_input("Take a picture")
NST = NeuralStyleTransfer()

style_options = np.array([style for style in NST.style_options.values()])

if img_file_buffer is not None:
    # To read image file buffer as a PIL Image:
    style_selection = st.selectbox(label="Select the desired painting style!",
                                   options=style_options)