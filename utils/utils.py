import cv2
import numpy as np
from utils.image_testing import print_image

RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH = 80, 160

def resize(image):
    """
    Resize the image to the input_image shape used by the network model
    """
    return cv2.resize(image, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT), cv2.INTER_AREA)

def rgb2yuv(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

def normalize(image):
    """
    Normalize the image from all possible domains into [-1, 1]
    """
    max_val = image.max()
    min_val = image.min()
    
    if max_val >= 1.0 and max_val <= 255.0:
        image = image / 127.5 - 1.0
    elif min_val >= 0.0 and max_val <= 1.0:
        image = image * 2 - 1.0
    else:
        image = 2 * (image-min_val)/(max_val - min_val) - 1

    return image

def preprocess(image):
    """
    Combine all preprocess functions into one
    """
    image = np.array(image)

    image_0 = (image*255).astype('uint8') # ! In this way, input image has to be [0,1]

    # pre-normalize to [0, 1]:
    # if max_val >= 1.0 and max_val <= 255.0:
    #     image = image/127.5 - 1.0
    # elif min_val >= 0 and max_val<= 1.0:
    #     image = image*2 - 1.0
    # else:
    #     image = 2 * (image-min_val)/(max_val - min_val) - 1

    image_resize = resize(image_0)
    image_yuv = rgb2yuv(image_resize)
    image_nor = normalize(image_yuv)
    
    # print_image(image_0, image_resize, image_yuv, image_nor)

    return image_nor