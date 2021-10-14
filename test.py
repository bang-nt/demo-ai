# Copyright (C) 2018-2021 coneypo
# SPDX-License-Identifier: MIT

# Author:   coneypo
# Blog:     http://www.cnblogs.com/AdaminXie
# GitHub:   https://github.com/coneypo/Dlib_face_recognition_from_camera
# Mail:     coneypo@foxmail.com

# Extract features from images and save into "features_all.csv"

import os
import dlib
import csv
import numpy as np
import logging
import cv2
from shutil import copyfile
from tkinter import Tk  # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askdirectory


class DataExtract:
    def __init__(self,
                 data_face_image_path='',
                 features_path='pkg_source/data/',
                 landmarks_path='pkg_source/data/data_dlib/shape_predictor_68_face_landmarks.dat',
                 resnet_model_path="pkg_source/data/data_dlib/dlib_face_recognition_resnet_model_v1.dat"
                 ):

        # Path of cropped faces
        self.data_face_image_path = data_face_image_path
        self.model_extract_path = os.path.abspath(os.path.join(data_face_image_path, os.pardir))

        # Path of features file
        self.features_path = features_path

        # Use frontal face detector of Dlib
        self.detector = dlib.get_frontal_face_detector()

        # Get face landmarks
        self.predictor = dlib.shape_predictor(landmarks_path)

        # Dlib Resnet Use Dlib resnet50 model to get 128D face descriptor
        self.face_reco_model = dlib.face_recognition_model_v1(resnet_model_path)

    # Delete old face folders
    def pre_work_del_old_face_folders(self):
        # "/data_faces_from_camera/person_x/"...
        if os.path.isfile(self.features_path + "/features_all.csv"):
            os.remove(self.features_path + "/features_all.csv")

        if os.path.isfile(self.features_path + "/names_all.csv"):
            os.remove(self.features_path + "/names_all.csv")

    # Return 128D features for single image
    # Input:    path_img           <class 'str'>
    # Output:   face_descriptor    <class 'dlib.vector'>
    def return_128d_features(self, path_img):
        img_rd = cv2.imread(path_img)
        faces = self.detector(img_rd, 1)

        logging.info("%-40s %-20s", "Image with faces detected:", path_img)

        # For photos of faces saved, we need to make sure that we can detect faces from the cropped images
        if len(faces) != 0:
            shape = self.predictor(img_rd, faces[0])
            face_descriptor = self.face_reco_model.compute_face_descriptor(img_rd, shape)
        else:
            face_descriptor = 0
            logging.warning("no face")
        return face_descriptor

    # Return the mean value of 128D face descriptor for person X
    # Input:    path_face_personX        <class 'str'>
    # Output:   features_mean_personX    <class 'numpy.ndarray'>
    def return_features_mean_personX(self, path_face_personX):
        features_list_personX = []
        photos_list = os.listdir(path_face_personX)
        if photos_list:
            for i in range(len(photos_list)):
                # Get 128D features for single image of personX
                logging.info("%-40s %-20s", "Reading image:", path_face_personX + "/" + photos_list[i])
                features_128d = self.return_128d_features(path_face_personX + "/" + photos_list[i])
                # Jump if no face detected from image
                if features_128d == 0:
                    i += 1
                else:
                    features_list_personX.append(features_128d)
        else:
            logging.warning("Warning: No images in%s/", path_face_personX)

        # Compute the mean
        # personX 的 N 张图像 x 128D -> 1 x 128D
        if features_list_personX:
            features_mean_personX = np.array(features_list_personX).mean(axis=0)
        else:
            features_mean_personX = np.zeros(128, dtype=int, order='C')
        return features_mean_personX

    def features_extract(self):
        self.pre_work_del_old_face_folders()
        # Get the order of latest person
        person_list = os.listdir(self.data_face_image_path)
        # person_num_list = []
        person_name_list = []
        for person in person_list:
            # person_num_list.append(int(person.split('_')[-1]))
            person_name_list.append(person)
        person_cnt = len(person_name_list)

        with open(self.model_extract_path + "/features_all.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for person in range(person_cnt):
                # Get the mean/average features of face/personX, it will be a list with a length of 128D
                logging.info("%s%s", self.data_face_image_path, person_name_list[person])

                features_mean_personX = self.return_features_mean_personX(
                    self.data_face_image_path + person_name_list[person])
                writer.writerow(features_mean_personX)
                logging.info('\n')
            logging.info(f"Save all the features of faces registered into:{self.model_extract_path}features_all.csv")

        with open(self.model_extract_path + "/names_all.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for person in range(person_cnt):
                logging.info("%s%s", self.data_face_image_path, person_name_list[person])

                person_name = person_name_list[person].lower()
                writer.writerow([person_name])
                logging.info('\n')
            logging.info(f"Save all the names of people registered into: {self.model_extract_path}names_all.csv\n\n")

        self.pre_work_del_old_face_folders()
        copyfile(self.model_extract_path + "/features_all.csv", self.features_path + "/features_all.csv")
        copyfile(self.model_extract_path + "/names_all.csv", self.features_path + "/names_all.csv")

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
    de = DataExtract(
        features_path='data/',
        landmarks_path='data/data_dlib/shape_predictor_68_face_landmarks.dat',
        resnet_model_path="data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

    filename = de.get_data_askfolder()
    try:
        de.features_extract()
    except Exception as e:
        print('Extract features on "{}" error.\nPlease try again and good luck!'.format(filename))
