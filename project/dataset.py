import shutil

import cv2
from os import listdir, unlink, remove
from os.path import isfile, join, exists, islink, isdir
import random

input_training_path = "dataset/training/input"
expected_training_path = "dataset/training/expected"

input_testing_path = "dataset/testing/input"


def CreateTiledSet(targetSize, windowShift, fileName, isNormal):
    img = cv2.imread(fileName)

    if img is None:
        raise ValueError(fileName + " cannot be opened!")
    if img.shape[0] % targetSize != 0 or img.shape[1] % targetSize != 0:
        raise ValueError(fileName + " cannot be tiled with " + str(targetSize) + "!")

    count = 0
    if isNormal:
        count = len([f for f in listdir(expected_training_path) if isfile(join(expected_training_path, f))])
    else:
        count = len([f for f in listdir(input_training_path) if isfile(join(input_training_path, f))])
    x = 0
    while x < img.shape[0] - targetSize:
        y = 0
        while y < img.shape[1] - targetSize:
            part = img[x:x + targetSize, y:y + targetSize, :]
            if isNormal:
                path = expected_training_path + "/normal_" + str(count) + ".png"
            else:
                path = input_training_path + "/rgb_" + str(count) + ".png"
            cv2.imwrite(path, part)

            count += 1
            y += windowShift
        x += windowShift


def CheckFlag():
    return exists("dataset/flag")


def ClearFolder(folder):
    for filename in listdir(folder):
        file_path = join(folder, filename)
        try:
            if isfile(file_path) or islink(file_path):
                unlink(file_path)
            elif isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def CreateDataset(targetSize=32, windowShift=4, clearIfExists=False):
    if CheckFlag():
        if not clearIfExists:
            print("Dataset creating will be skipped, because dataset is already exists!")
            return
        print("Deleting existing images...")
        ClearFolder(input_training_path)
        ClearFolder(expected_training_path)
        remove("dataset/flag")
        print("Files deleted!")

    original_rgbs = "images/rgb"
    original_normals = "images/normal"

    rgb_files = [f for f in listdir(original_rgbs) if isfile(join(original_rgbs, f))]
    normal_files = [f for f in listdir(original_normals) if isfile(join(original_normals, f))]

    for img in rgb_files:
        full_path = join(original_rgbs, img)
        CreateTiledSet(targetSize, windowShift, full_path, False)
    for img in normal_files:
        full_path = join(original_normals, img)
        CreateTiledSet(targetSize, windowShift, full_path, True)
    open("dataset/flag", 'w').close()


def GetRandomTrainingImages(max_count):
    rgb_img = None
    normal_img = None

    while rgb_img is None or normal_img is None:
        selected_num = random.randint(0, max_count)

        rgb_path = input_training_path + "/rgb_" + str(selected_num) + ".png"
        normal_path = expected_training_path + "/normal_" + str(selected_num) + ".png"

        rgb_img = cv2.imread(rgb_path)
        normal_img = cv2.imread(normal_path)

    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    normal_img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2RGB)
    return rgb_img, normal_img
