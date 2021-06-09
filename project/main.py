from he import he
from dhe import dhe
from ying import Ying_2017_CAIP

import numpy as np
import matplotlib.pyplot as plt
import imageio
import scipy, scipy.misc, scipy.signal
from skimage import exposure as ex

import cv2
import sys
import os

def main():
    path_dir = './test_images'
    file_list = os.listdir(path_dir)

    for file in file_list:
        img_name = path_dir +'/'+ file
        img = imageio.imread(img_name)
        
        he_img = he(img)
        dhe_img = dhe(img)
        ying_img =Ying_2017_CAIP(img)

        he_img = cv2.cvtColor(he_img, cv2.COLOR_RGBA2BGR)
        dhe_img = cv2.cvtColor(dhe_img, cv2.COLOR_RGBA2BGR)
        ying_img = cv2.cvtColor(ying_img, cv2.COLOR_RGBA2BGR)

        cv2.imwrite('./result_images/he/he_'+file, he_img)
        cv2.imwrite('./result_images/dhe/dhe_'+file, dhe_img)
        cv2.imwrite('./result_images/ying/ying_'+file, ying_img)


if __name__ == '__main__':
    main()