import cv2
import dlib
import numpy as np
import _pickle as pkl
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

import boto3
resource_s3 = boto3.resource('s3')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(resource_s3.Object('capstonefaceimg', "shape_predictor_68_face_landmarks.dat"))
ypr_model = load_model(resource_s3.Object('capstonefaceimg','model.h5'))


def return_std_model():
    x, y = pkl.load(open("samples.pkl", 'rb'))
    roll, pitch, yaw = y[:, 0], y[:, 1], y[:, 2]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

    std = StandardScaler()
    std.fit(x_train)
    return std

def train_model():
    x, y = pkl.load(open("samples.pkl", 'rb'))
    roll, pitch, yaw = y[:, 0], y[:, 1], y[:, 2]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

    std = return_std_model()
    x_train = std.transform(x_train)
    x_val = std.transform(x_val)
    x_test = std.transform(x_test)

    BATCH_SIZE = 64
    EPOCHS = 100
    model = Sequential()
    model.add(Dense(units=20, activation='relu', kernel_regularizer='l2', input_dim=x.shape[1]))
    model.add(Dense(units=10, activation='relu', kernel_regularizer='l2'))
    model.add(Dense(units=3, activation='linear'))

    #print(model.summary())

    callback_list = [EarlyStopping(monitor='val_loss', patience=25)]

    model.compile(optimizer='adam', loss='mean_squared_error')
    hist = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), batch_size=BATCH_SIZE, epochs=EPOCHS,
                     callbacks=callback_list)
    model.save('model.h5')

    # print('Train loss:', model.evaluate(x_train, y_train, verbose=0))
    # print('  Val loss:', model.evaluate(x_val, y_val, verbose=0))
    # print(' Test loss:', model.evaluate(x_test, y_test, verbose=0))



# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(os.path.join(DATA_PATH, "shape_predictor_68_face_landmarks.dat"))

def detect_face_points(image):
    # detector = dlib.get_frontal_face_detector()
    # predictor = dlib.shape_predictor(os.path.join(DATA_PATH, "shape_predictor_68_face_landmarks.dat"))
    face_rect = detector(image, 1)
    if len(face_rect) != 1: return []

    dlib_points = predictor(image, face_rect[0])
    face_points = []
    for i in range(68):
        x, y = dlib_points.part(i).x, dlib_points.part(i).y
        face_points.append(np.array([x, y]))
    return face_points


def compute_features(face_points):
    assert (len(face_points) == 68), "len(face_points) must be 68"

    face_points = np.array(face_points)
    features = []
    for i in range(68):
        for j in range(i + 1, 68):
            features.append(np.linalg.norm(face_points[i] - face_points[j]))

    return np.array(features).reshape(1, -1)


def return_ypr_score(main_ypr_model):
    img_array = np.fromfile(resource_s3.Object('capstonefaceimg','load/data/5-celebrity-faces-dataset/val/temp/test.jpg', np.uint8))
    im = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    face_points = detect_face_points(im)

    for x, y in face_points:
        cv2.circle(im, (x, y), 1, (0, 255, 0), -1)

    std = return_std_model()
    features = compute_features(face_points)
    features = std.transform(features)

    y_pred = main_ypr_model.predict(features)

    roll_pred, pitch_pred, yaw_pred = y_pred[0]

    roll_pred = round(roll_pred, 2)
    pitch_pred = round(pitch_pred, 2)
    yaw_pred = round(yaw_pred, 2)


    result_ypr = []
    result_ypr.append(roll_pred)
    result_ypr.append(pitch_pred)
    result_ypr.append(yaw_pred)

    return result_ypr

#return_ypr_score()
