import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import time
from tensorflow_examples.models.pix2pix import pix2pix


IMG_SIZE = 384
OUTPUT_CHANNELS = 7

depth_model = tf.keras.applications.MobileNetV2(input_shape=[IMG_SIZE, IMG_SIZE, 3], include_top=False)

# Use the activations of these layers
layer2_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    #'block_16_project',      # 4x4
]
layer2 = [depth_model.get_layer(name).output for name in layer2_names]
depth_stack = tf.keras.Model(inputs=depth_model.input, outputs=layer2)

base_model = tf.keras.applications.MobileNetV2(input_shape=[IMG_SIZE, IMG_SIZE, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
layers = [base_model.get_layer(name).output for name in layer_names]
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, 3])
    x = inputs

    inputs2 = tf.keras.layers.Input(shape=[IMG_SIZE,IMG_SIZE,3])
    x2 = inputs2


    # Downsampling through the model
    skips = down_stack(x) #tf.keras.Model(inputs=tf.keras.layers.Input(shape=[128, 128, 3]), outputs=layers)
    x = skips[-1]
    skips = reversed(skips[:-1]) # 'block_13_expand_relu',  # 8x8 'block_6_expand_relu',# 16x16 'block_3_expand_relu',   # 32x32   'block_1_expand_relu', # 64x64 

    #for depth stack
    skips2 = depth_stack(x2)

    skips2 = reversed(skips2[:])


    # Upsampling and establishing the skip connections
    for up, skip, skip2 in zip(up_stack, skips, skips2):
        x = up(x) #x initially is output of downstack i.e'block_16_project', layer then x becomes pix2pix.upsample(512, 3)(x) subsequently this x is the reslt of the last line of the previous iteration
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip]) # 1) concat(pix2pix.upsample(512, 3)('block_16_project'), 'block_13_expand_relu') (2)
        x= concat([x,skip2])
        


    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 3, strides=2,
        padding='same')  #64x64 -> 128x128

    x = last(x)
    return tf.keras.Model(inputs=[inputs, inputs2], outputs=x)

model = unet_model(OUTPUT_CHANNELS)
model.load_weights("./trained_weights/depth_weights/train_depth_2.h5")


image_rgb = cv2.imread(os.path.join("images","demo_images","aachen_000022_000019_leftImg8bit.png"))
image_depth = cv2.imread(os.path.join("images","demo_images","aachen_000022_000019_disparity.png"))
label = cv2.imread(os.path.join("images","demo_images","aachen_000022_000019_gtFine_polygons_encoding.png"),0)

img1 = image_rgb[:825, :1024]
depth1 = image_depth[:825, :1024]
label1 = label[:825, :1024]

img2 = image_rgb[:825, 1024:]
depth2 = image_depth[:825, 1024:]
label2 = label[:825, 1024:]

def segment(img):
    size = (np.shape(img[0])[1],np.shape(img[0])[0])
    img_c = cv2.resize(img[0],(IMG_SIZE,IMG_SIZE))
    img_d = cv2.resize(img[1],(IMG_SIZE,IMG_SIZE))
    img_c= img_c/255
    img_d = img_d/255
    prediction = model.predict((np.array([img_c]),np.array([img_d])))
    pred_img = tf.argmax(prediction, axis =-1)
    pred_img = pred_img[..., tf.newaxis][0]
    pred_img = cv2.resize(np.uint8(pred_img), size)

    return(pred_img)

input_1 = (img1,depth1)
input_2 = (img2,depth2)
pred_img1 = segment(input_1)
pred_img2 = segment(input_2)

ls_present = []
ls_present.append([img1,depth1,pred_img1,label1])
ls_present.append([img2,depth2,pred_img2,label2])

def present(ls):
    expand_ls =[]
    rows =len(ls)
    columns = len(ls[0])
    for i in ls_present:
        for j in i:
            expand_ls.append(j)
    fig = plt.figure(figsize=(15,20*rows))
    for k in range(rows*columns):
            fig.add_subplot(rows, columns, k+1)
            plt.imshow(expand_ls[k])
            
    plt.show()

present(ls_present)


