import random

from cnn import CellularNetwork, TrainingStep, ImageToCell
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import tensorflow as tf


def SplitImages(targetSize):
    rgbPath = "images/rgb"
    normalPath = "images/normal"
    files = [f for f in listdir(rgbPath) if isfile(join(rgbPath, f))]
    CheckFiles(files, False, rgbPath, targetSize)
    files = [f for f in listdir(normalPath) if isfile(join(normalPath, f))]
    CheckFiles(files, True, normalPath, targetSize)


def CheckFiles(files, isNormal, path, targetSize):
    for f in files:
        full_path = join(path, f)
        img = cv2.imread(full_path)
        if img.shape[0] % targetSize != 0 or img.shape[1] % targetSize != 0:
            print("Image " + f + " cannot be divided by " + str(targetSize) + ", skipping...")
            continue
        SaveImageParts(img, targetSize, isNormal)


def SaveImageParts(img, targetSize, isNormal):
    x = 0
    y = 0
    count = 0
    path = "images/"
    if isNormal:
        path += "normal/parts/normal_"
    else:
        path += "rgb/parts/RGB_"
    while x < img.shape[0]:
        while y < img.shape[1]:
            new_img = img[x:x + targetSize, y:y + targetSize, :]
            cv2.imwrite(path + str(count) + ".png", new_img)
            x += targetSize
            y += targetSize
            count += 1


def CreateModel():
    cnn = CellularNetwork()
    cnn.SetZ(0)
    cnn.SetA(np.ones((3, 3)))
    cnn.SetB(np.ones((3, 3)))

    return cnn


def TrainModel(model: CellularNetwork):
    for img in range(30):
        selected = random.randint(0, 31)
        input_img = "images/rgb/parts/RGB_" + str(selected) + ".png"
        expected_img = "images/normal/parts/normal_" + str(selected) + ".png"

        model.SetInput(input_img)
        model.SetState(np.zeros(model.Input.shape))
        expected_img = ImageToCell(cv2.cvtColor(cv2.imread(expected_img), cv2.COLOR_BGR2GRAY))
        expected_img = np.reshape(expected_img, [1, expected_img.shape[0], expected_img.shape[1], 1]).astype('float32')
        print("IMG " + str(img))
        for i in range(10):
            step = TrainingStep(model, expected_img)
            print("Loss: " + str(step))


cnn = CreateModel()
TrainModel(cnn)
