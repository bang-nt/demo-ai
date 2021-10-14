
# Extract features from images and save into "features_all.npy"

import os

import cv2
import numpy as np
import onnx
import onnxruntime.backend as backend

import logging

from shutil import copyfile
from tkinter import Tk  # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askdirectory
from pkg_source.util import postprocess, preprocess2


class DataExtract:
    def __init__(self,
                 data_face_image_path='',
                 features_path='pkg_source/data/',
                 onnx_model_path='pkg_source/models/FaceNet_vggface2_optmized.onnx'
                 ):

        # Path of cropped faces
        self.data_face_image_path = data_face_image_path
        self.model_extract_path = os.path.abspath(os.path.join(data_face_image_path, os.pardir))

        # Path of features file
        self.features_path = features_path

        self.onnx_model_path = onnx_model_path

    def img_inference(self, img_ori):
        # preprocess img
        img = preprocess2(img_ori)

        # inference
        onnx_model = onnx.load(self.onnx_model_path)
        ort_session = backend.prepare(onnx_model)
        features = ort_session.run(img)

        # postprogcess
        features = np.array(features[0])
        features = postprocess(features)

        return features

    # Delete old face folders
    def pre_work_del_old_face_folders(self):
        # "/data_faces_from_camera/person_x/"...
        if os.path.isfile(self.features_path + "/features_all.npy"):
            os.remove(self.features_path + "/features_all.npy")

        if os.path.isfile(self.features_path + "/names_all.txt"):
            os.remove(self.features_path + "/names_all.txt")

    def features_extract(self):
        # Get the order of latest person
        person_name_list = os.listdir(self.data_face_image_path)

        feature_list = []
        feature_list_name = []

        for label in person_name_list:
            print("start collecting faces from %s's data" % label)
            # ../data_faces_from_camera/bvnXXX/

            for img_name in os.listdir(self.data_face_image_path + '/' + label):
                # ../data_faces_from_camera/bvnXXX/image*.jpg
                img_path = self.data_face_image_path + '/' + label + '/' + img_name
                # print(img_path)
                feature_list_name.append(label)

                crop_img = cv2.imread(img_path)
                feature = self.img_inference(crop_img)
                feature_list.append(feature)

        feature_list = np.array(feature_list)
        np.save(self.model_extract_path + "/features_all.npy", feature_list)

        logging.info(f"Save all the features of people registered into: {self.model_extract_path}features_all.npy \n\n")

        with open(self.model_extract_path + "/names_all.txt", "w", newline="") as txtfile:
            for name in feature_list_name:
                logging.info("%s%s\n", self.data_face_image_path, name)
                person_name = name.lower()
                txtfile.writelines(person_name + '\n')

            logging.info(f"Save all the names of people registered into: {self.model_extract_path}names_all.txt \n\n")

        self.pre_work_del_old_face_folders()
        copyfile(self.model_extract_path + "/features_all.npy", self.features_path + "/features_all.npy")
        copyfile(self.model_extract_path + "/names_all.txt", self.features_path + "/names_all.txt")

    def get_data_askfolder(self):
        Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing

        dir_path = str(askdirectory(title='Select data faces folder'))
        if len(dir_path) < 2:
            print('Train cancel.')
        else:
            if not dir_path.endswith('/'):
                dir_path += '/'
            self.data_face_image_path = dir_path
        return dir_path


if __name__ == '__main__':
    te = DataExtract()
    path = te.get_data_askfolder()
    de = DataExtract(
        data_face_image_path=path,
        features_path='data/',
        onnx_model_path='data/models/FaceNet_vggface2_optmized.onnx'
        )

    # filename = de.get_data_askfolder()
    try:
        de.features_extract()
    except Exception as e:
        print('Extract features on "{}" error.\nPlease try again and good luck!'.format(path))
