from os import listdir
from os.path import join, isfile
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2 as cv


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

def show_set(data_line):
    HIST_BINS = np.linspace(0, 256, 16)
    for i in range(len(data_line[1:])):
        image = data_line[i+1]
        if image is None:
            continue
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        plt.subplot(2,4, i+1)
        plt.imshow(image, cmap='grey')
        plt.subplot(2,4, 4+i+1)
        plt.hist(image.ravel(),256,[0,256], histtype='bar')
        mean_hist = np.full((200000), int(np.mean(image.ravel())))
        plt.hist(mean_hist,128,[0,256], histtype='bar', color='red')
    plt.show()

def light_seg(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray,230,255,cv.THRESH_BINARY)
    image_processed = image.copy()
    image_processed[thresh == 255] = [0,0,0]

    plt.subplot(3,1,1)
    plt.imshow(gray, cmap='gray')
    plt.subplot(3,1,2)
    plt.imshow(thresh, cmap='gray')
    plt.subplot(3,1,3)
    plt.imshow(image_processed, cmap='gray')
    plt.show()

    return thresh, image_processed

def light_levels_seg(image, levels = 5):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    step = int(255 / levels)
    result = []
    counter = 1

    for i in range(0, 255, step):
        print(f"({i},{i+step})")
        _, lower_mask = cv.threshold(gray,i,     255,cv.THRESH_BINARY)
        _, upper_mask = cv.threshold(gray,i+step,255,cv.THRESH_BINARY)
        level_diff_mask = upper_mask - lower_mask
        result.append(level_diff_mask)
        plt.subplot(1,levels,counter)
        plt.imshow(level_diff_mask, cmap='gray')
        counter+=1

    plt.show()
    return result

if __name__ == '__main__':
    data = load_dataset("light_volume_dataset")
    data_table = pd.DataFrame(data)
    data_table = data_table.rename(columns={0:'name', 1:'dawn', 2:'noon', 3:'dusk', 4:'night'})
    print(data_table["name"])
    # Priority: 3,4 (with nightvision mode)

    #mask, processed_image = light_seg(data_table.iloc[3]["night"])
    processed_image = light_levels_seg(data_table.iloc[3]["night"])

    #show_set(data_table.iloc[1])
    #show_set(data_table.iloc[8])
    #show_set(data_table.iloc[4])
    #show_set(data_table.iloc[5])
    #show_set(data_table.iloc[6])
