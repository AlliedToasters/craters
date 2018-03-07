
# coding: utf-8

# In[1]:


import numpy as np
from PIL import Image
import os
from random import choice
import matplotlib.pyplot as plt
from generate_masks import pad_mask


# In[42]:


base_path = './tiles/keras_folders/'

def image_data_generator(path=base_path, train=True, batch_size=10, shuffle=False):
    """Custom image data generator for this problem."""
    if train:
        path += 'train/'
    else:
        path += 'test/'
    filenames = [x[:-4] for x in os.listdir(path)]
    ind = 0
    while True:
        x, y = [], []
        while len(x) < batch_size:
            if ind == len(filenames):
                ind = 0
            if shuffle:
                id_no = choice(filenames)
            else:
                id_no = filenames[ind]
            x_array = np.array(Image.open(path+'{}.png'.format(id_no)))/255
            x.append(np.expand_dims(x_array, axis=-1))
            y_array = np.array(Image.open(path[:-1]+'_mask/{}_mask.png'.format(id_no)))/255
            y.append(np.expand_dims(y_array, axis=-1))
            ind += 1
        x = np.array(x)
        y = np.array(y)
        yield x, y
    


# In[63]:


if __name__ == '__main__':
    gen = image_data_generator(shuffle=False, train=False)
    x1, y1 = next(gen)
    fig, ax = plt.subplots()
    ax.imshow(x1[7,:,:,0], cmap='Greys');
    ax.imshow(pad_mask(y1[7,:,:,0]), alpha=.2);

