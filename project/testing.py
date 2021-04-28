from typing import List

import cv2
import numpy as np
from tqdm import tqdm

from cnn import CellularNetwork, ImageToCell, CellToImage
from dataset import GetSeperatedTestRGB


def CreateNormalFromRGB(models: List[CellularNetwork], imageSize=32):
    imgs, shape = GetSeperatedTestRGB(imageSize)

    normals = []
    for img in tqdm(range(len(imgs))):
        gray = cv2.cvtColor(imgs[img], cv2.COLOR_BGR2GRAY)
        gray = ImageToCell(np.reshape(gray, [1, imageSize, imageSize, 1])).astype('float32')

        for i in range(3):
            models[i].SetInputAndState(gray, gray)

        red = models[0].Simulate(True)
        green = models[1].Simulate(True)
        blue = models[2].Simulate(True)

        red = CellToImage(np.reshape(red, [imageSize, imageSize]))
        green = CellToImage(np.reshape(green, [imageSize, imageSize]))
        blue = CellToImage(np.reshape(blue, [imageSize, imageSize]))

        norm = np.dstack((blue, green, red))
        normals.append(norm)

    fullPicture = None
    rows = shape[1] // imageSize
    cols = shape[0] // imageSize
    counter = 0

    for r in range(rows):
        rowImage = None
        for c in range(cols):
            if rowImage is None:
                rowImage = normals[counter]
            else:
                rowImage = np.concatenate((rowImage, normals[counter]), axis=1)
            counter += 1
        if fullPicture is None:
            fullPicture = rowImage
        else:
            fullPicture = np.concatenate((fullPicture, rowImage), axis=0)

    cv2.imwrite("final_normal.png", fullPicture)
