import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import cv2 as cv
import os
import gc

def get_dir_images(img_dir, img_shape, img_mode, dtype='uint8'): # e.g. img_dir="../DIC-C2DH-HeLa-Train/01", img_shape=(512, 512, 1), img_mode=cv.IMREAD_UNCHANGED
    print(f"Now reading: {img_dir}")
    img_list = os.listdir(img_dir)
    X = np.zeros((len(img_list),) + img_shape, dtype=dtype)
    for i, img_path in enumerate(img_list):
        img = cv.imread(f"{img_dir}/{img_path}", img_mode)  
        if img.shape != img_shape:
            img = img.reshape(img_shape)
        X[i] += img.astype(dtype)
    print("Done.")
    return X

def get_dir_masks(mask_dir, mask_shape, mask_mode, dtype='int32'): # e.g. img_dir="../DIC-C2DH-HeLa-Train/01", img_shape=(512, 512, 1), img_mode=cv.IMREAD_UNCHANGED
    print(f"Now reading: {mask_dir}")
    mask_list = os.listdir(mask_dir)
    y = np.zeros((len(mask_list),) + mask_shape, dtype=dtype)
    for i, mask_path in enumerate(mask_list):
        mask = (cv.imread(f"{mask_dir}/{mask_path}", mask_mode) != 0)
        if mask.shape != mask_shape:
            mask = mask.reshape(mask_shape)
        y[i] += mask.astype(dtype)
    print("Done.")
    return y

base_dir = './DIC-C2DH-HeLa-'

img_shape, img_mode = (512, 512, 1), cv.IMREAD_UNCHANGED
mask_shape, mask_mode = (512, 512), cv.IMREAD_UNCHANGED

train_subdirs = ['01', '01_ERR_SEG', '02', '02_ERR_SEG']
test_subdirs = ['01', '02']

data_dict = {'Train': dict.fromkeys(train_subdirs), 
             'Test': dict.fromkeys(test_subdirs)}

for key in data_dict.keys():
    for key_ in data_dict[key].keys():
        if 'ERR_SEG' not in key_:
            data_dict[key][key_] = get_dir_images(f"{base_dir}{key}/{key_}", img_shape, img_mode, 'uint8')
        else:
            data_dict[key][key_] = get_dir_masks(f"{base_dir}{key}/{key_}", mask_shape, mask_mode, 'int32')

train_01 = tf.data.Dataset.from_tensor_slices({'image': data_dict['Train'][train_subdirs[0]], 'mask': data_dict['Train'][train_subdirs[1]]})
train_02 = tf.data.Dataset.from_tensor_slices({'image': data_dict['Train'][train_subdirs[2]], 'mask': data_dict['Train'][train_subdirs[3]]})

test_01 = tf.data.Dataset.from_tensor_slices({'image': data_dict['Test'][test_subdirs[0]]})
test_02 = tf.data.Dataset.from_tensor_slices({'image': data_dict['Test'][test_subdirs[1]]})

print("Creating .tfrecord.")
tr_builder=  tfds.dataset_builders.store_as_tfds_dataset(name='hela', 
                                                         version='1.0.0',
                                                         config='train',
                                                         features=tfds.features.FeaturesDict({'image': tfds.features.Tensor(shape=img_shape, dtype=np.uint8), 
                                                                                              'mask': tfds.features.Tensor(shape=mask_shape, dtype=np.int32)}),
                                                         data_dir='TFRecords',
                                                         split_datasets={'01': train_01, 
                                                                         '02': train_02})
tr_builder.download_and_prepare()
print(".tfrecord created.")

print("Creating .tfrecord.")
te_builder=  tfds.dataset_builders.store_as_tfds_dataset(name='hela', 
                                                         version='1.0.0',
                                                         config='test',
                                                         features=tfds.features.FeaturesDict({'image': tfds.features.Tensor(shape=img_shape, dtype=np.uint8)}),
                                                         data_dir='TFRecords',
                                                         split_datasets={'01': test_01, 
                                                                         '02': test_02})
te_builder.download_and_prepare()
print(".tfrecord created.")