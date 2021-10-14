import dlib
import numpy as np
import cv2
import os
import logging
import math
import onnx
import onnxruntime.backend as backend
import ultra_light
from util import preprocess2, postprocess

BLINK_RATIO_THRESHOLD = 5.7

left_eye_landmarks = [36, 37, 38, 39, 40, 41]
right_eye_landmarks = [42, 43, 44, 45, 46, 47]


class FaceRecognizer:
    def __init__(self,
                 onnx_model_path='pkg_source/data/models/ultra_light_640_optimized.onnx',
                 facenet_model_path='pkg_source/data/models/FaceNet_vggface2_optmized.onnx',
                 data_features_path='pkg_source/data/features_all.npy',
                 data_names_path='pkg_source/data/names_all.txt'
                 ):
        self.font = cv2.FONT_ITALIC

        # load the model, create runtime session & get input variable name
        # onnx_model = onnx.load('models/ultra_light_640.onnx')
        self.detection_session = backend.prepare(
            onnx.load(onnx_model_path)
        )

        # onnx prepare
        self.facenet_session = backend.prepare(
            onnx.load(facenet_model_path)
        )

        # Save the features of faces in the database
        self.face_features_known_list = []
        # Save the name of faces in the database
        self.face_name_known_list = []

        # for data
        self.data_features_path = data_features_path
        self.data_names_path = data_names_path

    @staticmethod
    def midpoint(point1, point2):
        return (point1.x + point2.x) / 2, (point1.y + point2.y) / 2

    @staticmethod
    def euclidean_distance(point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def get_blink_ratio(self, eye_points, facial_landmarks):

        # loading all the required points
        corner_left = (facial_landmarks.part(eye_points[0]).x,
                       facial_landmarks.part(eye_points[0]).y)
        corner_right = (facial_landmarks.part(eye_points[3]).x,
                        facial_landmarks.part(eye_points[3]).y)

        center_top = self.midpoint(facial_landmarks.part(eye_points[1]),
                                   facial_landmarks.part(eye_points[2]))
        center_bottom = self.midpoint(facial_landmarks.part(eye_points[5]),
                                      facial_landmarks.part(eye_points[4]))

        # calculating distance
        horizontal_length = self.euclidean_distance(corner_left, corner_right)
        vertical_length = self.euclidean_distance(center_top, center_bottom)

        ratio = 0 if vertical_length == 0 else (horizontal_length / vertical_length)

        return ratio

    # Get known faces from "features_all.npy" and known names from "names_all.txt"
    def get_face_database(self):
        # print('Get face data')
        if os.path.exists(self.data_features_path) and os.path.exists(self.data_names_path):
            path_features_known_csv = self.data_features_path
            path_names_known_csv = self.data_names_path
            names = open(path_names_known_csv).readlines()
            features_list = np.load(path_features_known_csv).squeeze()
            if len(names) != len(features_list):
                logging.error("line numbers in 'features_all.csv' and 'names_all.csv' do not match!")
                return 0

            self.face_features_known_list = features_list
            self.face_name_known_list = names

            logging.info("Faces in Database： %d", len(self.face_features_known_list))
            return 1
        else:
            logging.warning("'features_all.npy' or 'names_all.txt' not found!")
            logging.warning("Please run 'Train model' to make and load model!")
            return 0

    @staticmethod
    # Compute the e-distance between two 128D features
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    # Face detection and recognition wit OT from input video stream
    def face_reco_with_blink_detect(self, img):
        """
        :param img: a cv2-like image
        :return: image processed; recognized names; recognized face (map with name list);
        blinkers (map with name list)
        """

        # 1. Get faces and names known from "features_all.csv" and "names_all.csv"
        if len(self.face_name_known_list) == 0:
            if not self.get_face_database():
                print('No faces found in database. Face Reco exit.')
                return

        frame = img.copy()
        h, w, _ = frame.shape

        # preprocess
        img = ultra_light.preprocess(frame)
        # inference
        confidences, boxes = self.detection_session.run(img)
        # postprocess
        boxes, labels, probs = ultra_light.postprocess(w, h, confidences, boxes, 0.6)

        # return list

        box_faces = []
        current_frame_face_name_list = []
        is_blinking = []

        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            x1, y1, x2, y2 = box

            # draw boxes
            crop_img = frame[y1:y2, x1:x2]
            if crop_img.shape[0] == 0 or crop_img.shape[1] == 0:
                continue

            predictions = "unknown"
            box_faces.append(box)

            # facenet inference
            img = preprocess2(crop_img)
            features1 = self.facenet_session.run(img)
            features1 = np.array(features1[0])
            features1 = postprocess(features1)

            # compare
            diff = np.subtract(self.face_features_known_list, features1)
            dist = np.sum(np.square(diff), axis=1)
            idx = np.argmin(dist)
            if dist[idx] < 1:  # schdule = ?
                predictions = self.face_name_known_list[idx].strip('\n')
                # print(predictions.strip('\n'))

                print(dist[idx], predictions, idx)

            # draw img
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            # cv2.rectangle(frame, (x1, y2 - 20), (x2, y2), (80, 18, 236), cv2.FILLED)
            text = predictions
            current_frame_face_name_list.append(text)
            cv2.putText(frame, text,
                        (x1 + 6, y2 - 6),
                        self.font, 0.8, (0, 255, 255),
                        1, cv2.LINE_AA)

        return frame, box_faces, current_frame_face_name_list
