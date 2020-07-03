import json
import kaggle
import zipfile
import pandas as pd
import albumentations as A
import cv2
from pathlib import Path, PurePath
from albumentations import (VerticalFlip, HorizontalFlip, Flip, RandomRotate90, Rotate, ShiftScaleRotate, CenterCrop, OpticalDistortion, GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
                            RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise, CLAHE, ChannelShuffle, InvertImg, RandomGamma, ToGray, PadIfNeeded 
                           )
from matplotlib import pyplot as plt
import os
import numpy as np
import tensorflow as tf
import shutil
from tensorflow.keras.utils import Sequence

#%%
def group_boxes(row):
    return row['bbox'].str[1:-1].str.split(',', expand=True).values.astype(float)

#load_image and change size


class DataGrenerator(Sequence):
    def __init__(self, df_bboxes, img_path, batch_size, augment, new_img_shape, shuffle):
        self.df_bboxes = df_bboxes
        self.img_path = img_path
        self.batch_size = batch_size
        self.augment = augment
        self.image_names = list(pd.Series(os.listdir(img_path)).str[:-4]) #remove.jpg
        self.bbox_params = self.get_bbox_params()
        self.new_img_shape = new_img_shape
        self.bbox_cols = ['x_min', 'y_min', 'bbox_width', 'bbox_height']
        self.aug_rules = self.augment_rules()
        self.image_grid = self.form_image_grid()
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def load_img(self, image_id):
        img = cv2.imread(train_path + image_id + '.jpg') / 255
        img = cv2.resize(img, self.new_img_shape)
        return img
    
    def augment_rules(self):
      aug_rules = A.Compose([
                        HorizontalFlip(p=0.5),
                        VerticalFlip(p=0.5),
                        A.RandomSizedBBoxSafeCrop(256, 256, p=1)
                        ],
                      p=1, bbox_params = self.bbox_params)
      return aug_rules
  
    def get_bbox_params(self):
        bbox_params = A.BboxParams(
          format = 'coco',
          label_fields = ['labels']
        )
        return bbox_params
    
    def get_bboxes(self, image_id):
        bboxes = self.df_bboxes[self.df_bboxes['image_id'] == image_id].loc[:, self.bbox_cols].values
        return bboxes

    def augment_img(self, img, bboxes):
      aug_result = self.aug_rules(image = img, bboxes = bboxes, labels = np.ones(len(bboxes)))
      return aug_result
  
    def get_aug_img_bboxes(self, aug_result):
        bbox_color = (1.0, 0, 0)
        bbox_thickness = 2
        img = aug_result['image']
        for bbox in aug_result['bboxes']:
            bbox = [int(i) for i in bbox]
            start_point = (bbox[0], bbox[1])
            end_point = (bbox[0] + bbox[2], bbox[1] + bbox[3])
            img = cv2.rectangle(img, start_point, end_point, bbox_color, bbox_thickness)
        return img
    
    def __data_generation(self, batch_ids):
        X, y = [], []
        if self.augment:
            for image_id in batch_ids:
                bboxes = self.get_bboxes(image_id)
                img = self.load_img(image_id)
                aug_result = self.augment_img(img, bboxes)
                label_grid = self.form_label_grid(aug_result['bboxes'])
                X.append(aug_result['image'])
                y.append(label_grid)
                print(image_id)
        else:
            for image_id in batch_ids:     
                img = self.load_img(image_id)
                bboxes = self.get_bboxes(image_id)
                label_grid = self.form_label_grid(bboxes)
                X.append(img)
                y.append(label_grid)
                
        return np.array(X), np.array(y)
    
    def __getitem__(self, index):        
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]        
        list_IDs_temp = [self.image_names[ii] for ii in indexes]        
        X, y = self.__data_generation(list_IDs_temp)
        return X, y
        
    def on_epoch_end(self):
      self.indexes = np.arange(len(self.image_names))
      if self.shuffle == True:
          np.random.shuffle(self.indexes)
          
    def __len__(self):
      return int(np.floor(len(self.image_names) / self.batch_size))
    
    def form_image_grid(self):    
        image_grid = np.zeros((10, 10, 4))
    
        # x, y, width, height
        cell = [0, 0, 256 / 10, 256 / 10] 
        for i in range(0, 10):
            for j in range(0, 10):
                image_grid[i,j] = cell
    
                cell[0] = cell[0] + cell[2]
    
            cell[0] = 0
            cell[1] = cell[1] + cell[3]
        return image_grid
    
    def form_label_grid(self, bboxes):
        label_grid = np.zeros((10, 10, 5))
    
        for i in range(0, 10):
            for j in range(0, 10):
                cell = self.image_grid[i,j]
                label_grid[i,j] = self.get_yolo_array(cell, bboxes)
    
        return label_grid
    
    def get_yolo_array(self, cell, bboxes):
        for bbox in bboxes:
            yolo_array = self.check_bbox_in_cell_and_return_yolo_array(cell, bbox)
            if yolo_array is not None:
                return yolo_array
        return [0 for i in range(5)]
        
    def check_bbox_in_cell_and_return_yolo_array(self, cell, bbox):
        cell_x, cell_y, cell_width, cell_height = cell
        cell_x_max = cell_x + cell_width
        cell_y_max = cell_y + cell_height
        bbox_x, bbox_y, bbox_width, bbox_height = bbox
        bbox_x_centre = bbox_x + bbox_width / 2
        bbox_y_centre = bbox_y + bbox_height / 2
        
        if(bbox_x_centre >= cell_x and 
        bbox_x_centre < cell_x_max and 
        bbox_y_centre >= cell_y and 
        bbox_y_centre < cell_y_max):
            yolo_array = self.create_yolo_array(bbox, cell)
            return yolo_array
        else:
            return None
        
    def create_yolo_array(self, bbox, cell):
        bbox_x, bbox_y, bbox_width, bbox_height = bbox
        cell_x, cell_y, cell_width, cell_height = cell
        
        bbox_x_centre = bbox_x + bbox_width / 2
        bbox_y_centre = bbox_y + bbox_height / 2
        
        bbox_x_centre_scaled = (bbox_x_centre - cell_x) / cell_width
        bbox_y_centre_scaled = (bbox_y_centre - cell_y) / cell_height
        
        bbox_width_scaled = bbox_width / 256
        bbox_height_scaled = bbox_height / 256
        
        yolo_array = [1, bbox_x_centre_scaled, bbox_y_centre_scaled,
                      bbox_width_scaled, bbox_height_scaled]
        return yolo_array
        
def get_bbox_cols(df, bbox_cols):
    df_bbox = df['bbox'].str[1:-1].str.split(',', expand=True).astype(float)
    df_bbox = df_bbox / 4
    df_bbox = df_bbox.astype(np.int32)
    df_bbox.columns = bbox_cols
    df = pd.concat([df, df_bbox], axis = 1)
    return df


#%%
os.chdir('/Users/joeodonnell-lyons/Desktop/git/wheat_detection')

df = pd.read_csv('train.csv')
bbox_cols = ['x_min', 'y_min', 'bbox_width', 'bbox_height']

df = get_bbox_cols(df, bbox_cols)

#rounding errors mean no bbox so we sort it out - could also just remove
df['bbox_width'] = df['bbox_width'].apply(lambda x: 1 if x==0 else x)
df['bbox_height'] = df['bbox_height'].apply(lambda x: 1 if x==0 else x) 



train_path = 'train/'
val_path = 'val/'

new_img_shape = (256, 256)
image_id = 'b6ab77fd7'

imgs = pd.Series(os.listdir(train_path))
val_imgs = imgs.iloc[3000:].copy()
for img_name in val_imgs:
    os.rename(os.path.join(train_path[:-1], img_name), os.path.join(val_path, img_name))
    shutil.move(os.path.join(train_path[:-1], img_name), os.path.join(val_path, img_name))
    os.replace(os.path.join(train_path[:-1], img_name), os.path.join(val_path, img_name))


train_DG = DataGrenerator(df, train_path, 1, True, new_img_shape, False)
val_DG = DataGrenerator(df, val_path, 1, False, new_img_shape, False)
X, y = train_DG[10]
test = train_DG[10]

def plot_img_with_img_index_boxes(X, y):
    image = np.squeeze(X).copy()
    y = np.squeeze(y)
    bbox_color = (1, 0, 0)
    bbox_thickness = 2
    for i in range(0, 10):
        for j in range(0, 10):            
            if y[i,j,0] >= 0.5:
                start_point = (int(j * 25.6), int(i * 25.6))
                end_point = (int((j + 1) * 25.6), int((i + 1) * 25.6))
                image = cv2.rectangle(image, start_point, end_point, bbox_color, bbox_thickness)
    
    plt.imshow(image)

#%% Model


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same', input_shape=(256, 256, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
    
    tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
    
    tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
    
    tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
    
    tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
    
    tf.keras.layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
    
    tf.keras.layers.Conv2D(1024, (3, 3), strides=(2, 2), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1)),
    
    ##### output layers #####
    
    tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.MaxPooling2D((1, 1), strides=(1, 1)),
    
    tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.MaxPooling2D((1, 1), strides=(1, 1)),
    
    tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.MaxPooling2D((1, 1), strides=(1, 1)),
    
    tf.keras.layers.Conv2D(5, (5, 5), strides=(1, 1), activation='sigmoid')
])
model.compile()
model.summary()


def custom_loss(y_true, y_pred):
    prob_loss = tf.keras.losses.binary_crossentropy(
        y_true[:,:,:,0], 
        y_pred[:,:,:,0]
    )
    
    bboxes_mask = tf.where(
        y_true[:,:,:,0] == 0, 
        0.5, 
        5.0
    )
    
    xy_loss = tf.keras.losses.MSE(
        y_true[:,:,:,1:3], 
        y_pred[:,:,:,1:3]
    )
    
    wh_loss = tf.keras.losses.MSE(
        y_true[:,:,:,3:5], 
        y_pred[:,:,:,3:5]
    )
    
    xy_loss = xy_loss * bboxes_mask
    wh_loss = wh_loss * bboxes_mask
    
    return prob_loss + xy_loss + wh_loss

optimiser = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(
    optimizer=optimiser, 
    loss=custom_loss
)

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=2, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1, restore_best_weights=True),
]


test = DataGrenerator(df, train_path, 1, True, new_img_shape, False)
for ii in range(3000):
    X, y = test[ii]


history = model.fit(
    DataGrenerator(df, train_path, 1, True, new_img_shape, False),
    #validation_data=val_DG,
    epochs=10,
    callbacks=callbacks
)

tf.__version__


