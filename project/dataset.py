import shutil
from typing import List, Tuple

import cv2
from os import listdir, unlink, remove
from os.path import isfile, join, exists, islink, isdir
import random

import numpy as np

input_training_path = "dataset/training/input"
expected_training_path = "dataset/training/expected"

input_testing_path = "dataset/testing/input"


def CreateTiledSet(targetSize: int, windowShift: int, fileName: str, isNormal: bool):
    """
    Creates tiles for training purposes from an image. I.e.: Makes a cut from the original image,
    with the size of the targetSize parameter, then moves the window by windowShift, creates another image, etc.
    @param targetSize: The size of the created training images
    @param windowShift: The distance of between the cutting windows
    @param fileName: The file to be processed
    @param isNormal: Is the picture a normal image or a RGB one?
    """
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


def CheckFlag() -> bool:
    """
    Checks if the training dataset is exists.
    @return: A bool depending on the dataset's existence
    """
    return exists("dataset/flag")


def ClearFolder(folder: str):
    """
    Deletes the contents of a folder
    @param folder: The path of the folder
    """
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
    """
    Creates the training dataset.
    @param targetSize: The size of the training images to be created
    @param windowShift: The distance in pixels between the cut images
    @param clearIfExists: If there is already a generated dataset, this function will stop.
    Set this to true, if you want to re-generate the dataset.
    @return: None
    """
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


def GetRandomTrainingImages(max_count=10000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Selects an RGB and normal image randomly, for training.
    @param max_count: The maximum value for random generation. Do not use, will be removed.
    @return: A randomly selected RGB and normal image
    """
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


def GetSeparatedTestRGB(target_size=32) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Selects a random RGB image from the testing folder and creates target sized parts from it.
    @param target_size: The size of the cut images
    @return: A list of the separated images and the size of the original image in a tuple
    """
    test_files = [f for f in listdir(input_testing_path) if isfile(join(input_testing_path, f))]
    random_file = join(input_testing_path, random.choice(test_files))

    img = cv2.imread(random_file)
    if img is None:
        raise ValueError("Cannot open " + random_file)

    parts = []
    x = 0
    while x < img.shape[0]:
        y = 0
        while y < img.shape[1]:
            part = img[x:x + target_size, y:y + target_size, :]
            parts.append(part)

            y += target_size
        x += target_size

    return parts, img.shape
