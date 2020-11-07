import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split


def load_image_data_2(folder_path, resize_row=None, resize_col=None, test_ratio=None, data_type='float32'):
    """
    Function for loading images and mask files. Resizing to target size and splitting based on test_ratio.
    :param folder_path: input folder address where the image and mask files are stored.
    :param resize_row: resize target row
    :param resize_col: resize target col
    :param test_ratio: ratio of images to be set aside for test dataset
    :param data_type: out data format
    :return: dictionary with x_train, y_train, x_test and y_test datasets
    """
    x = []
    y = []
    image_file_mat = [f for f in os.listdir(os.path.join(folder_path, 'image')) if
                      os.path.isfile(os.path.join(folder_path, 'image', f))]
    mask_file_mat = [f for f in os.listdir(os.path.join(folder_path, 'mask')) if
                     os.path.isfile(os.path.join(folder_path, 'mask', f))]
    files = list(set(image_file_mat).intersection(set(mask_file_mat)))
    for f in files:
        image_pil = Image.open(os.path.join(folder_path, 'image', f))
        mask_pil = Image.open(os.path.join(folder_path, 'mask', f))
        mask_pil = mask_pil.convert(mode='1')
        if image_pil.size[0] < image_pil.size[1]:
            image_pil = image_pil.rotate(90, expand=True)
            mask_pil = mask_pil.rotate(90, expand=True)
        if resize_row is not None and resize_col is not None:
            image = np.array(image_pil.resize((resize_col, resize_row)))
            mask = np.array(mask_pil.resize((resize_col, resize_row)))
        else:
            image = np.array(image_pil).astype(data_type)
            mask = np.array(mask_pil).astype(data_type)
        x.append(image)
        y.append(mask)
    x = np.array(x).astype(data_type)
    x = x / 255
    y = np.array(y).astype(data_type)
    y = np.expand_dims(y, -1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_ratio)
    print('Train image dataset shape: %s' % str(x_train.shape))
    print('Train label dataset shape: %s' % str(y_train.shape))
    print('Test image dataset shape: %s' % str(x_test.shape))
    print('Test label dataset shape: %s' % str(y_test.shape))
    print('-' * 70)
    return {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}


def load_image_data_1(folder_path, resize_row=None, resize_col=None, test_ratio=None, data_type='float32'):
    """
    Function for loading images and mask files. Resizing to target size and splitting based on test_ratio.
    :param folder_path: input folder address where the image and mask files are stored.
    :param resize_row: resize target row
    :param resize_col: resize target col
    :param test_ratio: ratio of images to be set aside for test dataset
    :param data_type: out data format
    :return: dictionary with x_train, y_train, x_test and y_test datasets
    """
    mask_files = os.listdir(os.path.join(folder_path, 'mask'))
    raw_files = os.listdir(os.path.join(folder_path, 'image'))
    # find intersection of two lists
    files = list(set(raw_files).intersection(mask_files))
    test_files = list(set(mask_files).symmetric_difference(raw_files))
    x_train = []
    x_test = []
    y_train = []
    for f in files:
        mask = Image.open(os.path.join(folder_path, 'mask', f))
        mask = np.array(mask.resize((resize_col, resize_row)))
        mask = mask.mean(axis=2)
        mask[mask < 250] = 0
        mask[mask > 0] = 1
        if not mask.sum():
            continue
        y_train.append(mask)
        raw = Image.open(os.path.join(folder_path, 'image', f))
        raw = np.array(raw.resize((resize_col, resize_row)))
        x_train.append(raw)
    for f in test_files:
        try:
            raw = Image.open(os.path.join(folder_path, 'image', f))
        except:
            continue
        raw = np.array(raw.resize((resize_col, resize_row)))
        x_test.append(raw)
    x_train = np.array(x_train).astype(data_type)
    x_train /= 255
    x_test = np.array(x_test).astype(data_type)
    x_test /= 255
    y_train = np.array(y_train)
    y_train = np.expand_dims(y_train, 3)

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=test_ratio)
    print('Train image dataset shape: %s' % str(x_train.shape))
    print('Train label dataset shape: %s' % str(y_train.shape))
    print('Test image dataset shape: %s' % str(x_test.shape))
    print('Test label dataset shape: %s' % str(y_test.shape))
    print('-' * 70)
    return {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}
