from typing import List

import cv2
import numpy as np
from matplotlib import pyplot as plt

from cnn import CellularNetwork, TrainingWrapper, ImageToCell
from project.dataset import GetRandomTrainingImages


def StartTraining(models: List[CellularNetwork], skips: List[bool], iterations=1000, consoleInfo=100):
    red_loss = None
    green_loss = None
    blue_loss = None

    if not skips[0]:
        print("Training red model...")
        red_loss = TrainModel(models[0], 0, iterations, consoleInfo)
        red_avg = sum(red_loss) / len(red_loss)
        if models[0].ModelBestLoss > red_avg:
            print("Model was performing better: " + str(models[0].ModelBestLoss) + " to " + str(red_avg))
            models[0].ModelBestLoss = red_avg
            models[0].SaveNetwork("red_network")
    else:
        print("Skipping red model.")

    if not skips[1]:
        print("Training green model...")
        green_loss = TrainModel(models[1], 1, iterations, consoleInfo)
        green_avg = sum(green_loss) / len(green_loss)
        if models[1].ModelBestLoss > green_avg:
            print("Model was performing better: " + str(models[1].ModelBestLoss) + " to " + str(green_avg))
            models[1].ModelBestLoss = green_avg
            models[1].SaveNetwork("green_network")
    else:
        print("Skipping green model")

    if not skips[2]:
        print("Training blue model...")
        blue_loss = TrainModel(models[2], 2, iterations, consoleInfo)
        blue_avg = sum(blue_loss) / len(blue_loss)
        if models[2].ModelBestLoss > blue_avg:
            print("Model was performing better: " + str(models[2].ModelBestLoss) + " to " + str(blue_avg))
            models[2].ModelBestLoss = blue_avg
            models[2].SaveNetwork("blue_network")
    else:
        print("Skipping blue model")
    print("Training finished!")

    if red_loss is not None:
        plt.plot(red_loss, 'r-')
    if green_loss is not None:
        plt.plot(green_loss, 'g-')
    if blue_loss is not None:
        plt.plot(blue_loss, 'b-')
    plt.show()


def TrainModel(model: CellularNetwork, color: int, iterations=1000, consoleInfo=100):
    training_function = TrainingWrapper()
    losses = []

    for i in range(iterations):
        inp, exp = GetRandomTrainingImages(130000)
        inp = ImageToCell(np.reshape(cv2.cvtColor(inp, cv2.COLOR_RGB2GRAY), [1, 32, 32, 1])).astype('float32')
        exp = ImageToCell(np.reshape(exp[:, :, color], [1, 32, 32, 1])).astype('float32')

        model.SetInputAndState(inp, inp)

        last_loss = training_function(model, exp).numpy()
        losses.append(last_loss)
        if i % consoleInfo == 0 and consoleInfo != 0:
            print("Loss at iteration " + str(i) + ": " + str(last_loss))

    return losses
