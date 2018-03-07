
# coding: utf-8

# In[8]:


import tensorflow as tf
import keras
from keras import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate, Cropping2D, BatchNormalization
from keras.initializers import VarianceScaling
from build_image_generator import image_data_generator
import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


# In[2]:

init = VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=0)

inputs = Input(shape=(256, 256, 1))

normed_inputs = BatchNormalization()(inputs)
conv1 = Conv2D(filters = 64, kernel_size=(3, 3), activation='relu', kernel_initializer=init)(normed_inputs)
conv2 = Conv2D(filters = 64, kernel_size=(3, 3), activation='relu', kernel_initializer=init)(conv1)
normed1 = BatchNormalization()(conv2)
maxp1 = MaxPooling2D(pool_size=(2, 2))(normed1)

conv3 = Conv2D(filters = 128, kernel_size=(3, 3), activation='relu', kernel_initializer=init)(maxp1)
conv4 = Conv2D(filters = 128, kernel_size=(3, 3), activation='relu', kernel_initializer=init)(conv3)
normed2 = BatchNormalization()(conv4)
maxp2 = MaxPooling2D(pool_size=(2, 2))(normed2)

conv5 = Conv2D(filters = 256, kernel_size=(3, 3), activation='relu', kernel_initializer=init)(maxp2)
conv6 = Conv2D(filters = 256, kernel_size=(2, 2), activation='relu', kernel_initializer=init)(conv5)
normed3 = BatchNormalization()(conv6)
maxp3 = MaxPooling2D(pool_size=(2, 2))(normed3)


conv7 = Conv2D(filters = 512, kernel_size=(3, 3), activation='relu', kernel_initializer=init)(maxp3)
conv8 = Conv2D(filters = 512, kernel_size=(3, 3), activation='relu', kernel_initializer=init)(conv7)

up1 = Conv2DTranspose(filters = 256, kernel_size=(2, 2), strides=(2,2))(conv8)
crop1 = Cropping2D(cropping=(4,4))(normed3)
concat1 = Concatenate(axis=-1)([crop1, up1])
conv9 = Conv2D(filters = 256, kernel_size=(3, 3), activation='relu', kernel_initializer=init)(concat1)
conv10 = Conv2D(filters = 256, kernel_size=(3, 3), activation='relu', kernel_initializer=init)(conv9)

up2 = Conv2DTranspose(filters = 128, kernel_size=(2, 2), strides=(2,2))(conv10)
crop2 = Cropping2D(cropping=(15,15))(normed2)
concat2 = Concatenate(axis=-1)([crop2, up2])
conv11 = Conv2D(filters = 128, kernel_size=(3, 3), activation='relu', kernel_initializer=init)(concat2)
conv12 = Conv2D(filters = 128, kernel_size=(3, 3), activation='relu', kernel_initializer=init)(conv11)

up3 = Conv2DTranspose(filters = 64, kernel_size=(2, 2), strides=(2,2))(conv12)
crop3 = Cropping2D(cropping=(38,38))(normed1)
concat3 = Concatenate(axis=-1)([crop3, up3])
conv13 = Conv2D(filters = 64, kernel_size=(3, 3), activation='relu', kernel_initializer=init)(concat3)
conv14 = Conv2D(filters = 64, kernel_size=(3, 3), activation='relu', kernel_initializer=init)(conv13)
output = Conv2D(filters = 1, kernel_size=(1, 1), activation='sigmoid', kernel_initializer=init)(conv14)


# In[3]:


unet = Model(inputs=inputs, outputs=output)


# In[4]:


#print(unet.summary())


# In[5]:



def focal_loss(target, output, gamma=2):
    output /= K.sum(output, axis=-1, keepdims=True)
    eps = K.epsilon()
    output = K.clip(output, eps, 1. - eps)
    return -K.sum(K.pow(1. - output, gamma) * target * K.log(output),
                  axis=-1)


unet.compile(
    optimizer = Adam(amsgrad=True),
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)


# In[6]:

n_examples = 500
batch_size = 1
train_gen = image_data_generator(shuffle=True, batch_size=batch_size)
val_gen = image_data_generator(train=False, batch_size=batch_size)


# In[ ]:


checkpointer = ModelCheckpoint('./models/unet_first_model.hdf5', save_best_only=True, verbose=True)


# In[7]:




history = unet.fit_generator(
    train_gen,
    steps_per_epoch=n_examples//batch_size,
    epochs=10,
    verbose=True,
    validation_data=val_gen,
    validation_steps=1,
    callbacks=[checkpointer]
)

