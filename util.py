import cv2
import numpy as np


def Normalize(data):
    """function torh.nn.functional.normalize(data, p=2, dim=1)"""
    denom = [np.linalg.norm(data[i]) for i in range(len(data))]
    norm_data = [data[i] / denom[i] for i in range(len(data))]
    return np.array(norm_data)


def preprocess(img):
    if type(img) == str:
        img = cv2.imread(img)
    mean_vec = np.array([123.68, 116.28, 103.53])
    stddev_vec = np.array([57.6, 57.6, 57.6])
    img = cv2.resize(img, (160, 160))
    img = np.transpose(img, (2, 0, 1))
    norm_img_data = np.zeros(img.shape).astype('float32')
    for i in range(img.shape[0]):
        norm_img_data[i, :, :] = (img[i, :, :] - mean_vec[i]) / stddev_vec[i]
    norm_img_data = np.expand_dims(norm_img_data, axis=0).astype(np.float32)
    return norm_img_data


def preprocess2(img):
    if type(img) == str:
        img = cv2.imread(img)
    # print(img.shape)
    img = cv2.resize(img, (160, 160))
    img = np.transpose(img, (2, 0, 1))
    norm_img_data = np.zeros(img.shape).astype('float32')
    for i in range(img.shape[0]):
        norm_img_data[i, :, :] = (img[i, :, :] - 127.5) / 128
    norm_img_data = np.expand_dims(norm_img_data, axis=0).astype(np.float32)
    return norm_img_data


def postprocess(features):
    features = Normalize(features)
    return features
