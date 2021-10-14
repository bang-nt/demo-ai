import os
import sys
import time
from tkinter import Tk  # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askdirectory

import cv2
import numpy as np
import onnx
import onnxruntime.backend as backend

import ultra_light


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


def img_inference(img_ori):
    # preprocess img 
    img = preprocess2(img_ori)

    # inference
    onnx_model = onnx.load('models/FaceNet_vggface2_optmized.onnx')
    ort_session = backend.prepare(onnx_model)
    features = ort_session.run(img)

    # postprogcess
    features = np.array(features[0])
    features = postprocess(features)

    return features


def prepare_features_custom(video_dir):
    # ../data_faces_from_camera/

    id_list = os.listdir(video_dir)

    feature_list = []
    feature_list_name = []

    for label in id_list:

        print("start collecting faces from %s's data" % (label))
        # ../data_faces_from_camera/bvnXXX/
        print(os.listdir(video_dir + '/' + label))
        for img_name in os.listdir(video_dir + '/' + label):

            # ../data_faces_from_camera/bvnXXX/image*.jpg
            img_path = video_dir + '/' + label + '/' + img_name
            print(img_path)
            feature_list_name.append(label)

            crop_img = cv2.imread(img_path)
            feature = img_inference(crop_img)
            feature_list.append(feature)

    feature_list = np.array(feature_list)
    np.save("./features.npy", feature_list)

    f = open("./features.txt", "w+")
    for i in feature_list_name:
        f.writelines(i + "\r\n")
    f.close()

    print("save features.")


def get_data_askfolder():
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing

    dir_path = str(askdirectory(title='Select data faces folder'))
    if len(dir_path) < 2:
        print('Train cancel.')
    else:
        if not dir_path.endswith('/'):
            dir_path += '/'

    return dir_path


def demo_video(video_file=None):

    # load the model, create runtime session & get input variable name
    onnx_model = onnx.load('models/ultra_light_640_optimized.onnx')
    # onnx_model = onnx.load('models/ultra_light_640.onnx')
    detection_session = backend.prepare(onnx_model)

    # onnx prepare
    facenet_model = onnx.load('models/FaceNet_vggface2_optmized.onnx')
    facenet_session = backend.prepare(facenet_model)

    # video_capture = cv2.VideoCapture(0)
    video_capture = cv2.VideoCapture(video_file if video_file is not None else 0)
    # LIVE_URL = "rtsp://admin:admin123@172.16.1.29/cam/realmonitor?channel=1&subtype=0"
    # video_capture = cv2.VideoCapture(LIVE_URL)
    print("Video fps:", video_capture.get(cv2.CAP_PROP_FPS))

    names = open("./names_all.txt").readlines()
    features_list = np.load("./features_all.npy").squeeze()

    print(type(names), len(names), len(features_list))
    # exit()
    while True:
        ret, img_ori = video_capture.read()
        if ret == False:
            break
        # if img_ori is None:
        #     continue

        h, w, _ = img_ori.shape

        # preprocess
        img = ultra_light.preprocess(img_ori)
        # inference
        confidences, boxes = detection_session.run(img)
        # postprocess 
        boxes, labels, probs = ultra_light.postprocess(w, h, confidences, boxes, 0.6)

        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            x1, y1, x2, y2 = box
            predictions = "unknown"

            # draw boxes
            crop_img = img_ori[y1:y2, x1:x2]
            if crop_img.shape[0] == 0 or crop_img.shape[1] == 0:
                continue

            # facenet inference
            img = preprocess2(crop_img)
            features1 = facenet_session.run(img)
            features1 = np.array(features1[0])
            features1 = postprocess(features1)

            # compare
            diff = np.subtract(features_list, features1)
            dist = np.sum(np.square(diff), axis=1)
            idx = np.argmin(dist)
            if dist[idx] < 1:  # schdule = ?
                predictions = names[idx].strip('\n')
                # print(predictions.strip('\n'))

                print(dist[idx], predictions, idx)

            # draw img
            cv2.rectangle(img_ori, (x1, y1), (x2, y2), (255, 255, 255), 2)
            # cv2.rectangle(img_ori, (x1, y2 - 20), (x2, y2), (80, 18, 236), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            text = predictions
            cv2.putText(img_ori, text,
                        (x1 + 6, y2 - 6),
                        font, 0.8, (0, 255, 255),
                        1, cv2.LINE_AA)

        cv2.imshow('Video', img_ori)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    # path = get_data_askfolder()
    # prepare_features_custom(path)
    # demo_image()
    demo_video()