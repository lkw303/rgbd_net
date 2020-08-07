import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix


import matplotlib.pyplot as plt

import os
import numpy as np
import cv2






IMG_SIZE = 192
OUTPUT_CHANNELS = 7
EPOCHS = 1000
BATCH_SIZE = 64

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

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

#down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]
def unet_model(output_channels):

    inputs = tf.keras.layers.Input(shape=[IMG_SIZE,IMG_SIZE,3])
    x = inputs

    inputs2 = tf.keras.layers.Input(shape=[IMG_SIZE,IMG_SIZE,3])
    x2 = inputs2

    # Downsampling through the model
    skips = down_stack(x) #tf.keras.Model(inputs=tf.keras.layers.Input(shape=[128, 128, 3]), outputs=layers)
    x = skips[-1]
    skips = reversed(skips[:-1])

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

model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate = 0.000001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

weights_path = os.path.join("./trained_weights", "depth_weights","train_depth_2.h5")

model.load_weights(weights_path)

#===================================================================================
train_image_path = os.path.join(os.path.expanduser("~"), "Desktop", "cityscape","leftImg8bit","train")
train_depth_path = os.path.join(os.path.expanduser("~"), "Desktop", "cityscape","disparity_trainvaltest","disparity", "train")
train_label_path = os.path.join(os.path.expanduser("~"), "Desktop","cityscape", "gtFine_trainvaltest","gtFine","train_encoding")



X_train_img = []
X_train_depth = []
Y_train = []

def create_train_data():
    for subdir in os.listdir(train_label_path):
        city_label_folder_path = os.path.join(train_label_path,subdir)
        city_image_folder_path = os.path.join(train_image_path,subdir)
        city_depth_folder_path = os.path.join(train_depth_path,subdir)
        for files in os.listdir(city_label_folder_path):
            data_id = str(files)[:-len("_gtFine_polygons_encoding.png")]
            print(data_id)
            segmented_img = cv2.imread(os.path.join(os.path.join(city_label_folder_path,files)), 0)
            segmented_img = segmented_img[:825,:]
            photo_img = cv2.imread(os.path.join(city_image_folder_path,data_id + "_leftImg8bit.png"))
            photo_img = photo_img[:825,:]
            depth_img = cv2.imread(os.path.join(city_depth_folder_path,data_id + "_disparity.png"))
            depth_img = depth_img[:825,:]

            segmented_img = cv2.resize(segmented_img,(IMG_SIZE,IMG_SIZE))
            segmented_img = np.reshape(segmented_img,(IMG_SIZE,IMG_SIZE,1))
            photo_img= cv2.resize(photo_img, (IMG_SIZE,IMG_SIZE))/255
            depth_img = cv2.resize(depth_img, (IMG_SIZE,IMG_SIZE))/255

            X_train_img.append(photo_img)
            X_train_depth.append(depth_img)
            Y_train.append(segmented_img)




create_train_data()





X_train_img = np.reshape(X_train_img,(-1, IMG_SIZE,IMG_SIZE,3))
X_train_depth =np.reshape(X_train_depth,(-1, IMG_SIZE,IMG_SIZE,3))
X_train = (X_train_img,X_train_depth)

Y_train = np.reshape(Y_train, (-1,IMG_SIZE,IMG_SIZE,1))


#===================================================================================

model_history = model.fit(X_train, Y_train, 
                        batch_size = BATCH_SIZE,
                        epochs = 1000, 
                        validation_split =0.1,
                        verbose =1,
                        )

model.save_weights(os.path.join("./", "trained_weights", "train_depth_3.h5"))
model.save(os.path.join("./", "trained_weights", "train_depth_full_model_1.h5"))

predict1 = model.predict(np.expand_dims(X_train[0], axis =0))

plt.imshow(np.squeeze(predict1))
plt.show()
