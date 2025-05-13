from os import listdir
from os.path import join, isfile
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2 as cv


# Utility
def load_dataset(path):
    new_path = join(path, "noon")
    files = [f for f in listdir(new_path) if isfile(join(new_path, f))]
    result = []
    for file in files:
        pack = []
        for rel_path in ["dawn", "noon", "dusk", "night"]:
            image = cv.imread(join(path, rel_path, file))
            pack += [image]
        result += [[file] + pack]
    return result

# Utility
def light_mean(image):
    return light_mean_grey(cv.cvtColor(image, cv.COLOR_BGR2GRAY))

# Utility
def light_mean_grey(image_grey):
    mean = int(np.mean(image_grey.ravel()))
    return mean

# Utility
def light_mean_masked(image_grey):
    array_image = image_grey.ravel()
    array_image = array_image[array_image != 0]
    if len(array_image) == 0:
        return 0
    return int(np.mean(array_image))

def show_set(data_line):
    """
    Shows set images and their histograms
    """

    #HIST_BINS = np.linspace(0, 256, 16)
    for i in range(len(data_line[1:])):
        image = data_line[i+1]
        if image is None:
            continue
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        plt.subplot(2,4, i+1)
        plt.imshow(image, cmap='grey')
        plt.subplot(2,4, 4+i+1)
        plt.hist(image.ravel(),256,[0,256], histtype='bar')
        mean_hist = np.full((200000), light_mean(image))
        plt.hist(mean_hist,128,[0,256], histtype='bar', color='red')
    plt.show()

def light_seg(image, level=230):
    """
    <in>
    level - parameter, 0-255. All pixels with
    value lower than this will be reduced to 0
    
    <out>
    thresh - mask, with black and white pixels
    image_processed - image with applied mask
    """


    gray = image
    if len(image.shape)>1:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray,level,255,cv.THRESH_BINARY)

    return thresh

def light_levels_seg(image, levels = 5):
    """
    Divides image into masks
    Each mask shows pixels of its range
    
    e.g. 2 levels will divide image into 2 masks
    mask1 - pixel values in 0-127
    mask2 - pixel values in 128-255

    returns list of masks
    """
    

    gray = image
    if len(image.shape)>1:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    step = int(255 / levels)
    result = []
    counter = 1

    for i in range(0, 255, step):
        _, lower_mask = cv.threshold(gray,i,     255,cv.THRESH_BINARY)
        _, upper_mask = cv.threshold(gray,i+step,255,cv.THRESH_BINARY)
        level_diff_mask = upper_mask - lower_mask
        result.append(level_diff_mask)
        counter+=1

    return result

def detect(image, upper_bound=255):
    """
    Detects the level of brightness on image (0 to upper_bound int)

    upper_bound - parameter, pixel value euqal to this 
    and above are considered too bright and not counted
    """

    
    image_grey = image
    if len(image.shape)>1:
        image_grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_grey[image_grey == 0] = 1

    low_level_mask = light_seg(image, upper_bound)
    image_grey[low_level_mask != 0] = 0

    mean_value_masked = light_mean_masked(image_grey)

    return mean_value_masked #float(mean_value_masked) / 255.0

def detect_with_mask(image, mask, upper_bound=255):
    """
    Similar to detect

    mask parameter contains matrix, where every ignored pixel is 0
    mean value of image is calculated with other non-zero pixels

    upper_bound - parameter, pixel value euqal to this 
    and above are considered too bright and not counted
    """

    image_grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    low_level_mask = light_seg(image, upper_bound)
    image_grey[low_level_mask != 0] = 0
    image_grey[mask == 0] = 0

    mean_value_masked = light_mean_masked(image_grey)

    return mean_value_masked

if __name__ == '__main__':
    data = load_dataset("light_volume_dataset")
    data_table = pd.DataFrame(data)
    data_table = data_table.rename(columns={0:'name', 1:'dawn', 2:'noon', 3:'dusk', 4:'night'})
    print(data_table["name"])
    
    image = data_table.iloc[8]['night']
    print(detect(image))
