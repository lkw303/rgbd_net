import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import time

IMG_SIZE = 384
'''
train_data =[]

train_label_path = os.path.join(os.path.expanduser("~"),"Desktop","tensorflow_trials","cityscape_archive","gtFine_new", "train")
train_image_path = os.path.join(os.path.expanduser("~"),"Desktop","tensorflow_trials","cityscape_archive","leftImg8bit", "train")

def create_train_data():
    for subdir in os.listdir(train_label_path):
        city_label_folder_path = os.path.join(train_label_path,subdir)
        city_image_folder_path = os.path.join(train_image_path,subdir)
        for files in os.listdir(city_label_folder_path):
            data_id = str(files)[:20]
            print(data_id)
            segmented_img = cv2.imread(os.path.join(os.path.join(city_label_folder_path,files)), cv2.IMREAD_COLOR)
            photo_img = cv2.imread(os.path.join(city_image_folder_path,data_id + "_leftImg8bit.png"), cv2.IMREAD_COLOR)

            segmented_resize = cv2.resize(segmented_img,(IMG_SIZE,IMG_SIZE))
            photo_resize = cv2.resize(photo_img, (IMG_SIZE,IMG_SIZE))
            train_data.append([photo_resize,segmented_resize])

create_train_data()

X_train = []
Y_train = []

for features, label in train_data:
    X_train.append(features)
    Y_train.append(label)

'''
#label = cv2.imread("./cityscape_archive/gtFine_new/train_encoding/bremen/bremen_000000_000019_gtFine_polygons_encoding.png",0)
#img = cv2.imread("./cityscape_archive/leftImg8bit/train/bremen/bremen_000000_000019_leftImg8bit.png")

image = cv2.imread("./cityscape_archive/leftImg8bit/val/frankfurt/frankfurt_000001_007285_leftImg8bit.png")
label = cv2.imread("./cityscape_archive/gtFine_new/val_encoding/frankfurt/frankfurt_000001_007285_gtFine_polygons_encoding.png",0)

img1 = image[:825, :1024]
label1 = label[:825, :1024]

img2 = image[:825, 1024:]
label2 = label[:825, 1024:]

def segment(img):
    img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    img = img/255
    model = tf.keras.models.load_model("./trained_weights/train_encode_split_1.h5")
    start = time.time()
    prediction = model.predict(np.array([img]))
    pred_img = tf.argmax(prediction, axis =-1)
    pred_img = pred_img[..., tf.newaxis][0]

    return(pred_img)

pred_img1 = segment(img1)
pred_img2 = segment(img2)

ls_present = []
ls_present.append([img1,pred_img1,label1])
ls_present.append([img2,pred_img2,label2])

def present(ls):
    expand_ls =[]
    rows =len(ls)
    columns = 3
    for i in ls_present:
        for j in i:
            expand_ls.append(j)
    fig = plt.figure(figsize=(15,20*rows))
    for k in range(rows*columns):
            fig.add_subplot(rows, columns, k+1)
            plt.imshow(expand_ls[k])
            
    plt.show()

present(ls_present)

'''
fig = plt.figure(figsize =(15,5))
columns = 3
rows = 1

fig.add_subplot(1,3,1)
plt.imshow(img)

fig.add_subplot(1,3,2)
plt.imshow(pred_img)

fig.add_subplot(1,3,3)
plt.imshow(label)
'''

