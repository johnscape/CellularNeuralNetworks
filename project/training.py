from typing import List

import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm, trange

from cnn import CellularNetwork, TrainingWrapper, ImageToCell
from dataset import GetRandomTrainingImages


def StartTraining(models: List[CellularNetwork], skips: List[bool], iterations=1000, consoleInfo=100):
    red_loss = ModelStep(models[0], iterations, consoleInfo, 0, skips[0])
    green_loss = ModelStep(models[1], iterations, consoleInfo, 1, skips[1])
    blue_loss = ModelStep(models[2], iterations, consoleInfo, 2, skips[2])
    print("Training finished!")

    if red_loss is not None:
        plt.plot(red_loss, 'r-')
    if green_loss is not None:
        plt.plot(green_loss, 'g-')
    if blue_loss is not None:
        plt.plot(blue_loss, 'b-')
    plt.show()


def ModelStep(model: CellularNetwork, iterations: int, consoleInfo: int, color: int, skipping=False):
    trainingText = "Training "
    fileName = "_network"
    skippingText = "Skipping "

    if color == 0:
        trainingText += "red model..."
        fileName = "red" + fileName
        skippingText += "red model."
    elif color == 1:
        trainingText += "green model..."
        fileName = "green" + fileName
        skippingText += "green model."
    else:
        trainingText += "blue model..."
        fileName = "blue" + fileName
        skippingText += "blue model."

    loss = None
    if not skipping:
        print(trainingText)
        loss = TrainModel(model, color, iterations, consoleInfo)
        avg = sum(loss) / len(loss)
        if model.ModelBestLoss > avg:
            print("Model was performing better: " + str(model.ModelBestLoss) + " to " + str(avg))
            model.ModelBestLoss = avg
            model.SaveNetwork(fileName)
    else:
        print(skippingText)
    return loss


def TrainModel(model: CellularNetwork, color: int, iterations=1000, consoleInfo=100, imageSize=32):
    training_function = TrainingWrapper()
    losses = []
    barText = ""
    if color == 0:
        barText = "Training red network"
    elif color == 1:
        barText = "Training green network"
    else:
        barText = "Training blue network"
    tr = trange(iterations, desc=barText, leave=True)
    count = 0
    for i in tr:
        inp, exp = GetRandomTrainingImages(130000)
        inp = ImageToCell(
            np.reshape(cv2.cvtColor(inp, cv2.COLOR_RGB2GRAY), [1, imageSize, imageSize, 1])
        ).astype('float32')

        exp = ImageToCell(
            np.reshape(exp[:, :, color], [1, imageSize, imageSize, 1])
        ).astype('float32')

        model.SetInputAndState(inp, inp)

        last_loss = training_function(model, exp).numpy()
        losses.append(last_loss)
        if count % consoleInfo == 0 and consoleInfo != 0:
            newText = barText + ", loss at " + str(count) + ": " + str(last_loss)
            tr.set_description(newText)
            tr.refresh()

        count += 1

    return losses
