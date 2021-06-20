# -*- coding: utf-8 -*-


import random
from collections import defaultdict

import numpy as np
import os
from utils_tmp import *
import matplotlib.pyplot as plt

from datetime import datetime
from keras import backend as K
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import Model
from keras.preprocessing import image

if __name__ == '__main__':

    seed_dir ='./seeds_50'
    img_dir = './generated_inputs/NC3'
    img_paths = os.listdir(img_dir)
    img_paths.pop()
    img_paths.pop()

    img_num = len(img_paths)
    L2_norm=[]
    seed_imgs=[]
    for i in range(img_num):
        seed_name = img_paths[i].split('-')[2] + '.png'
        seed_path = os.path.join(seed_dir,seed_name)
        img_path = os.path.join(img_dir, img_paths[i])
        gen_img = preprocess_image(img_path)
        seed_imgs.append(seed_path)
        orig_img = preprocess_image(seed_path)
        diff_img = gen_img - orig_img
        L2_norm.append(np.linalg.norm(diff_img))
    num_seed = len(set(seed_imgs))
    print(num_seed)
    L2 = np.array(L2_norm)
    avg = np.mean(L2)
    print(avg)
    plt.hist(L2_norm)
    plt.show()
