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

    def transform_bulk(self, cols=3, saveArt=True):
        """ Retrieve and transform all content in content folder """
        style_type = self.style_options[int(input(self.style_options))]
        style_image_tensor = self.image_to_tensor(path_to_img=f"{self.style_path}\{style_type}",
                                                  img_size=self.style_img_size)
        images = []
        for file in os.listdir(self.content_path):
            content_image_tensor = self.image_to_tensor(path_to_img=f"{self.content_path}\{file}", 
                                                        img_size=self.content_img_size)
            combined_result = self.hub_module(tf.constant(content_image_tensor), 
                                              tf.constant(style_image_tensor))[0]
            images.append(self.tensor_to_image(tensor=combined_result, 
                                               plot_img=False))
            if saveArt:
                os.mkdir(f".\Output\{style_type.split(".")[0]}_{date.today}")
                _, ax = plt.subplots(figsize=(15, 15))
                ax.imshow(images[-1])
                ax.axis('off')
                plt.savefig(f'.\Output\{style_type.split(".")[0]}_{date.today}\{file}')
        
        nr_of_images = len(images)
        rows = math.ceil(nr_of_images / cols)

        # Plot style image
        print('Style Image:')
        fig, ax = plt.subplots(1)
        plt.imshow(plt.imread(f'.\Styles\{style_type}'))
        plt.axis('off')
        plt.show()

        # Plot output images
        print('Output Images:')
        _, axs = plt.subplots(rows, cols, figsize=(15, 15))
        for idx, img in enumerate(images):
            axs[idx//cols][idx%cols].imshow(img)
            axs[idx//cols][idx%cols].axis('off')

        plt.show()

    def transform_all_styles(self, content_name, print_style=True):
        """ Transform content in all available style types """
        content_image_tensor = self.image_to_tensor(path_to_img=f"{self.content_path}\{content_name}", 
                                                    img_size=self.content_img_size)
        
        # Loop over all style types and transform content image
        for idx, styleType in enumerate(list(self.style_options.values())):
            style_image_tensor = self.image_to_tensor(path_to_img=f"{self.style_path}\{styleType}",
                                                      img_size=self.style_img_size)
            combined_result = self.hub_module(tf.constant(content_image_tensor), 
                                              tf.constant(style_image_tensor))[0]
            image = self.tensor_to_image(tensor=combined_result, 
                                         plot_img=False)

            if print_style:
                # Create and plot image
                fig, ax = plt.subplots(1, 2, figsize=(15, 8))
                ax[0].imshow(plt.imread(f'.\Styles\{styleType}'))
                ax[0].axis('off')

                ax[1].imshow(image)
                ax[1].axis('off')
                fig.suptitle(f'---  {styleType.split(".")[0]}  ---')
                self.watermark(ax[1], fig)
                plt.savefig(f'{self.save_location}/Person_{self.save_cntr}-Art_{idx}')
                self.save_cntr += 1
                
            else:
                # Create and plot image
                fig, ax = plt.subplots(1, figsize=(10, 7.5))
                ax.imshow(image)
                ax.axis('off')
                fig.suptitle(f'---  {styleType.split(".")[0]}  ---')
                self.watermark(ax, fig, size_divided=4)
                plt.show(f'{self.save_location}/Person_{self.save_cntr}-Art_{idx}')
                self.save_cntr += 1
    
    def _RUN(self):
        """ Run the entire program, outputting new images when new photos are created """
        while True:
            imageFolder = set(os.listdir(self.content_path))
            if len(imageFolder) > self.img_cntr:
                print(f'Found new image to process')

                # If image is added, retrieve image name, transform and show
                img_path = list(imageFolder.difference(self.processed_content))[0]
                self.transform_all_styles(content_name=img_path)

                # Add processed image to list and increment counter
                self.processed_content = set(os.listdir(self.content_path))
                self.img_cntr += 1
            
            time.sleep(1)



NST = NeuralStyleTransfer()

# NST._RUN()

NST.transform_bulk()

