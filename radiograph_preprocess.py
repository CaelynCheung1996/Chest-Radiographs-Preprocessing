# Chest Radiograph Pre-processing
# Xiaohui Zhang
# July, 4th, 2019

import sys
import os
from os.path import isfile, join
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from skimage import data, img_as_float
from skimage import exposure
from skimage.io import imread
from skimage import filters
from scipy.misc import imsave, imfilter
from scipy import ndimage

import matplotlib.patches as mpatches
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import dilation, closing, square, disk, binary_dilation
from medpy.filter.smoothing import anisotropic_diffusion
from skimage.feature import canny 
from skimage.transform import resize

matplotlib.rcParams['font.size'] = 8


def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.

    """
    image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf

def bounding_box(img):
    rows = np.any(img, axis = 1)
    cols = np.any(img, axis = 0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def annotation_detector(img):
    # Sauvola text filter 
    window_size = 25  # must be odd number
    k = 0.5
    
    thresh_sauvola = filters.threshold_sauvola(img, window_size)
    binary_sauvola = img > thresh_sauvola
    label_sauvola = label(binary_sauvola)
        
    return binary_sauvola, label_sauvola

def triplize_img_channel(img):
    img_channel3 = np.repeat(img[..., np.newaxis], 3, -1)
    
    return img_channel3

def process_chest_radiographs(argv):
    # argv[0]: /path/to/original radiographs
    # argv[1]: /path/to/preprocessed radiographs
    
    raw_folder = argv[0]
    raw_imglist = []
    preprocess_folder = argv[1]
    preprocess_imglist = []

    # list image file
    if os.path.exists(raw_folder):
        raw_imglist = [file for file in os.listdir(raw_folder) if isfile(join(raw_folder,file))]
        raw_imglist.sort()

    for i in range(0,len(raw_imglist)): 

    # ======================================================================================= 
	# read images
        print "Loading Image {}".format(raw_imglist[i])
        raw_filename = raw_folder + '/' + raw_imglist[i]
        preprocess_filename = preprocess_folder + '/' + raw_imglist[i]
        image = imread(raw_filename)        
        #print"Image size: {}".format(image.shape)
    # =====================================================================================
	# binarize the image and get bounding box
        thresh = filters.threshold_otsu(image)
        bw = closing(image > thresh, disk(10))
        label_image = label(bw)

        # find bounding box
        # rmin, rmax, cmin, cmax = bounding_box(bw)
	    # crop image using bounding box
        # cropped_image = image[rmin:rmax, cmin:cmax]
 
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,6))
        for region in regionprops(label_image):
             # take regions with large enough areas
            if region.area >= 640000:
                # draw rectangle around segmented coins
                minr, minc, maxr, maxc = region.bbox
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
                # crop image wihin boudning box
                crop_image = image[minr:maxr, minc:maxc]
#                ax1.add_patch(rect)
#                ax1.imshow(image, cmap = 'gray')

    # ========================================================================================       
       #  resize image
        resize_shape = 2048
        resize_image = resize(crop_image, (resize_shape, resize_shape), anti_aliasing=True)
        
	# =======================================================================================
	# annoate the text
        binary_sauvola, label_sauvola = annotation_detector(resize_image)
        # ax1.imshow(binary_sauvola)
        clean_mask = np.zeros(resize_image.shape)

        for alphabet in regionprops(label_sauvola):
#            print"label: {}, area:{}".format(alphabet.label, alphabet.area)
            if alphabet.area > 300 and alphabet.area < 3500:
                    label_index = alphabet.label
                    label_mask = label_index * np.ones(resize_image.shape)
                    mask_sauvola = (label_sauvola == label_mask)
                    clean_mask += mask_sauvola

        clean_mask = binary_dilation(clean_mask, disk(5))
        #ax1.imshow(clean_mask, cmap = 'gray') 
        unblurred_image = img_as_float(resize_image) * (1-clean_mask)
        blurred_image = img_as_float(filters.median(resize_image, disk(50))) * clean_mask
        annofree_image = blurred_image + unblurred_image
     
       # =================================================================================
        # CLAHE
        image_adapteq = exposure.equalize_adapthist(annofree_image, clip_limit = 0.03) 
      
	# =================================================================================
        # anistropic denoising
        image_smoothed = anisotropic_diffusion(image_adapteq, niter = 5)
        
        # ================================================================================
        # canny edge detection
        #image_canny = canny(image_smoothed, sigma = 3) 
        #ax2.imshow(image_canny, cmap = 'gray') 
        # =================================================================================
        # resize image
        #ax1.set_axis_off
        #ax2.set_axis_off
        #plt.tight_layout() 
        #plt.show()
        
          
        imsave(preprocess_filename, image_smoothed)


if __name__ == '__main__':
    process_chest_radiographs(sys.argv[1:]) 


