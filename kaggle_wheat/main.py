import pandas as pd
import numpy as np
import tensorflow as tf
from DataGenerator import DataGenerator
from model import get_model_and_callbacks

def get_bbox_cols(df, bbox_cols):
    df_bbox = df['bbox'].str[1:-1].str.split(',', expand=True).astype(float)
    df_bbox = df_bbox / 4
    df_bbox = df_bbox.astype(np.int32)
    df_bbox.columns = bbox_cols
    df = pd.concat([df, df_bbox], axis=1)
    return df

if __name__ == '__main__':
    df = pd.read_csv('train.csv')
    bbox_cols = ['x_min', 'y_min', 'bbox_width', 'bbox_height']
    df = get_bbox_cols(df, bbox_cols)
    # rounding errors mean no bbox so we sort it out - could also just remove
    df['bbox_width'] = df['bbox_width'].apply(lambda x: 1 if x == 0 else x)
    df['bbox_height'] = df['bbox_height'].apply(lambda x: 1 if x == 0 else x)
    train_path = 'train/'
    val_path = 'val/'
    new_img_shape = (256, 256)

    train_DG = DataGenerator(df, train_path, 1, True, new_img_shape, False)
    val_DG = DataGenerator(df, val_path, 1, False, new_img_shape, False)
    X, y = train_DG[10]
    test = train_DG[10]
    
    model, callbacks = get_model_and_callbacks()

    history = model.fit(
        DataGenerator(df, train_path, 1, True, new_img_shape, False),
        validation_data=val_DG,
        epochs=10,
        callbacks=callbacks
    )