from typing import List

import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm, trange

from cnn import CellularNetwork, TrainingWrapper, ImageToCell
from dataset import GetRandomTrainingImages


def StartTraining(models: List[CellularNetwork], skips: List[bool], iterations=1000, consoleInfo=100, size=32):
    """
    Use this function to train your models.
    @param models: A list of three models in the following order:
    The model for the red channel, for the green and for the blue
    @param skips: A list of three bools. If any bool is True, the corresponding model's training will be skipped.
    Useful if you have already trained one or more models.
    @param iterations: How many images will be shown to each model.
    @param consoleInfo: At every consoleInfo-th iteration, a message will be displayed on the console with the
    latest loss value.
    @param size: The size of the training images.
    """
    red_loss = ModelStep(models[0], iterations, consoleInfo, 0, skips[0], size)
    green_loss = ModelStep(models[1], iterations, consoleInfo, 1, skips[1], size)
    blue_loss = ModelStep(models[2], iterations, consoleInfo, 2, skips[2], size)
    print("Training finished!")

    if red_loss is not None:
        plt.plot(red_loss, 'r-')
    if green_loss is not None:
        plt.plot(green_loss, 'g-')
    if blue_loss is not None:
        plt.plot(blue_loss, 'b-')
    plt.show()


def ModelStep(model: CellularNetwork, iterations: int, consoleInfo: int, color: int, skipping=False, imgSize=32):
    """
    An intermediate function for a more understandable code. Use StartTraining instead of this.
    @param model: The model to train.
    @param iterations: The images to show to the model.
    @param consoleInfo: The frequency of printing information to the console.
    @param color: The model's color channel (0 - red, 1 - green, 2 - blue)
    @param skipping: If set to True, training will be skipped
    @param imgSize: The size of the training images
    @return: None
    """
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
        loss = TrainModel(model, color, iterations, consoleInfo, imgSize)
        avg = sum(loss) / len(loss)
        if model.ModelBestLoss > avg:
            print("Model was performing better: " + str(model.ModelBestLoss) + " to " + str(avg))
            model.ModelBestLoss = avg
            model.SaveNetwork(fileName)
    else:
        print(skippingText)
    return loss


def TrainModel(model: CellularNetwork, color: int, iterations=1000, consoleInfo=100, imageSize=32):
    """
    This function handles model training. Use StartTraining instead of this.
    @param model: The model to train.
    @param color: The model's color channel (0 - red, 1 - green, 2 - blue)
    @param iterations: The images to show to the model.
    @param consoleInfo: The frequency of printing information to the console.
    @param imageSize: The size of the training images.
    @return: None
    """
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
        inp, exp = GetRandomTrainingImages(10000)  # TODO: read the number of images once, this could cause an error
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
