import cv2
import numpy as np
import matplotlib.pyplot as plt

def print_image(image_0, image_resize=None, image_yuv=None, image_nor=None):
    '''
    image_0: original image
    if only one input, the preprocess will be done in the function
    '''
    if image_resize is None:
        if isinstance(image_0, str): # if input is an image path, string
            
            #image = '/home/jiaqq/Documents/ase22/datasets/dataset1/track1/normal/IMG/center_2024_08_14_10_57_26_754.png'
            image_0 = plt.imread(image_0)
            #obs1_rgb = cv2.cvtColor(obs1, cv2.COLOR_BGR2RGB) # cv.imread read the image in BGR format; plt.imread read in RGB
        else:
            image_0 = np.array(image_0)    
            image_resize = cv2.resize(image_0, (160, 80), interpolation=cv2.INTER_AREA)
            image_yuv = cv2.cvtColor(image_resize, cv2.COLOR_RGB2YUV)
            
            from utils.utils import normalize
            image_nor = normalize(image_yuv)

    # subplots with 1 row and 4 columes
    fig, axarr = plt.subplots(4, 1, figsize=(5, 15))

    # first colume: original image
    axarr[0].imshow(image_0)
    axarr[0].set_title(f"Original Image\nMax value: {image_0.max()}")
    axarr[0].axis('off')  # Hide axes

    # 2nd colume: resized image
    axarr[1].imshow(image_resize)
    axarr[1].set_title(f"Resized Image\nMax value: {image_resize.max()}")
    axarr[1].axis('off')

    # 3rd colume: YUV image
    axarr[2].imshow(image_yuv)
    axarr[2].set_title(f"YUV Image\nMax value: {image_yuv.max()}")
    axarr[2].axis('off')

    # 4th colume: normalized image
    axarr[3].imshow(image_nor)
    axarr[3].set_title(f"YUV Image\nMax value: {image_nor.max()}\nMin value: {image_nor.min()}")
    axarr[3].axis('off')

    plt.show()