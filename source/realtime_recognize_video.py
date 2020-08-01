# because i didn't have any camera so i try in video
import numpy as np
import sklearn
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

import source.rawdata as raw

if __name__ == '__main__':
    # load dataset
    dataset = np.load('../5-celebrity-faces-dataset/5-celebrity-faces-embedding.npz')
    trainX, trainy, testX, testy = dataset['arr_0'], dataset['arr_1'], dataset['arr_2'], dataset['arr_3']
    # normalize input
    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(trainX)
    # label encode target
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    trainy = out_encoder.transform(trainy)
    # fit model
    model = SVC(kernel='linear', probability=True)
    model.fit(trainX, trainy)

    # open video
    ben_video = cv2.VideoCapture('../video/ben affleck/ben afleck.mp4')
    # read frame from video
    ret, frame = ben_video.read()
    # load model facenet
    facenet = tf.keras.models.load_model('../model/facenet_keras.h5')

    while ret:
        # get boxes and face extracted
        boxes, faces_array = raw.extract_faces(frame)
        faces_array = np.asarray(faces_array)
        # embedding face
        embedded_faces = []
        for i in range(len(faces_array)):
            embedded_face = faces_array[i]
            embedded_face = raw.get_embedding(facenet, embedded_face)
            embedded_faces.append(embedded_face)

        # predict face
        yhat_class = model.predict(embedded_faces)
        predict_name = out_encoder.inverse_transform(yhat_class)

        for i in range(len(boxes)):
            # get box
            x1, y1, x2, y2 = boxes[i]
            # draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 18, 236), 2)
            cv2.rectangle(frame, (x1, y2 - 20), (x2, y2), (80, 18, 236), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            # set name
            label = "face: {}".format(i)
            # input name of face
            cv2.putText(frame, label, (x1 + 6, y2 - 6), font, 0.5, (255, 255, 255), 1)

        cv2.imshow('frame', frame)
        ret, frame = ben_video.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
