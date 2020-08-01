from PIL import Image
import numpy as np
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import sklearn


# model = tf.keras.models.load_model('../model/facenet_keras.h5')

# extract a single face from given photo
def extract_face(filename, require_size=(160, 160)):
    # load image from file name
    image = Image.open(filename)
    # convert to 3 chanels if it hasn't
    image = image.convert('RGB')
    # convert to array
    pixels = np.asarray(image)
    # create detector, using default weights
    detector = MTCNN()
    # detect face in image
    results = detector.detect_faces(pixels)
    # get bounding box from first face
    x1, y1, width, height = results[0]['box']
    # get abs
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract face
    face = pixels[y1:y2, x1:x2]
    # resize image
    image = Image.fromarray(face)
    image = image.resize(require_size)
    face_array = np.asarray(image)
    return face_array


# load images and extract faces for all images in directory
def load_faces(directory):
    faces = list()
    # enumerate files
    for filename in os.listdir(directory):
        # get path file
        path = directory + filename
        # extract face
        face = extract_face(path)
        # store
        faces.append(face)
    return faces


# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
    # x for faces and y for labels
    x, y = list(), list()
    # enumerate directory
    for subdir in os.listdir(directory):
        # get path subdir
        path = directory + subdir + '/'
        # skip any files that might be in the dir
        if not os.path.isdir(path):
            continue
        # load all faces in the path
        faces = load_faces(path)
        # create label
        labels = [subdir for _ in range(len(faces))]
        # summarize
        print('loaded {} examples for class: {}'.format(len(faces), subdir))
        # store
        x.extend(faces)
        y.extend(labels)
    return np.asarray(x), np.asarray(y)


def get_embedding(model, face_pixels):
    # scales pixels values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face to one sample
    samples = np.expand_dims(face_pixels, axis=0)
    # make predict to get embedding
    yhat = model.predict(samples)
    return yhat[0]


if __name__ == '__main__':
    # load train dataset
    trainX, trainy = load_dataset('../5-celebrity-faces-dataset/train/')
    print(trainX.shape, trainy.shape)
    # load test dataset
    testX, testy = load_dataset('../5-celebrity-faces-dataset/val/')
    # load model
    model = tf.keras.models.load_model('../model/facenet_keras.h5')
    print('loaded model')
    # save face dataset
    np.savez_compressed('../5-celebrity-faces-dataset/5-celebrity-faces-dataset.npz', trainX, trainy, testX, testy)
    # convert each face in train set to embedding
    newTrain = list()
    for face_pixel in trainX:
        embedding = get_embedding(model, face_pixel)
        newTrain.append(embedding)
    newTrain = np.asarray(newTrain)
    print(newTrain.shape)
    # convert each image in test set to embedding
    newTest = list()
    for face_pixel in testX:
        embedding = get_embedding(model, face_pixel)
        newTest.append(embedding)
    newTest = np.asarray(newTest)
    print(newTest.shape)
    # save arrays to one file in compressed format
    np.savez_compressed('../5-celebrity-faces-dataset/5-celebrity-faces-embedding.npz', newTrain, trainy, newTest, testy)

