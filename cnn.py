import numpy as np
import cv2
from enum import Enum
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from os.path import exists


class BoundaryTypes(Enum):
    CONSTANT = 0,
    ZERO_FLUX = 1,
    PERIODIC = 2


def ImageToCell(image):
    image = (-1) * ((image.astype(np.float)) / 127.5 - 1.0)
    return image


def CellToImage(cell):
    cell = ((cell * -1) + 1.0) * 127.5
    return cell.astype(np.uint8)


class CellularNetwork:
    def __init__(self):
        initializer = tf.initializers.glorot_uniform()
        self.B = tf.Variable(initializer([3, 3, 1, 1]), dtype=tf.float32)
        self.A = tf.Variable(initializer([3, 3, 1, 1]), dtype=tf.float32)
        self.Z = tf.Variable(initializer([1]), dtype=tf.float32)

        self.Input = None
        self.State = None

        self.SimTime = 1
        self.TimeStep = 0.1

        self.Boundary = BoundaryTypes.CONSTANT
        self.BoundValue = 0

        self.Optimizer = tf.optimizers.Adam(0.1)

        self.ModelBestLoss = 10.0

    def GetOutput(self):
        return self.State

    def SetTimeStep(self, Ts):
        self.TimeStep = Ts

    def SetSimTime(self, T):
        self.SimTime = T

    def SetInputAndState(self, inp, stat):
        self.SetInput(inp)
        self.SetState(stat)

    def SetInput(self, In):
        if isinstance(In, str):
            tempImg = cv2.imread(In)
            if tempImg is None:
                raise ValueError("Input image cannot be loaded!")
            tempImg = ImageToCell(cv2.cvtColor(tempImg, cv2.COLOR_BGR2GRAY))
            tempImg = np.reshape(tempImg, [1, tempImg.shape[0], tempImg.shape[1], 1])
        else:
            tempImg = In

        if self.Input is None:
            self.Input = tf.Variable(tempImg, dtype=tf.float32)
        else:
            self.Input.assign(tempImg)

    def SetState(self, St):
        if isinstance(St, str):
            tempImg = cv2.imread(St)
            if tempImg is None:
                raise ValueError("Input image cannot be loaded!")
            tempImg = ImageToCell(cv2.cvtColor(tempImg, cv2.COLOR_BGR2GRAY))
            tempImg = np.reshape(tempImg, [1, tempImg.shape[0], tempImg.shape[1], 1])
        else:
            tempImg = St
        if self.State is None:
            self.State = tf.Variable(tempImg, dtype=tf.float32)
        else:
            self.State.assign(tempImg)

    def SetZ(self, z):
        self.SetBias(z)

    def SetBias(self, z):
        z = np.asarray([z])
        self.Z.assign(z)

    def SetA(self, a):
        self.SetATemplate(a)

    def SetATemplate(self, a):
        if a.shape != [3, 3, 1, 1]:
            a = np.reshape(a, [3, 3, 1, 1])
        self.A.assign(a)

    def SetB(self, a):
        self.SetBTemplate(a)

    def SetBTemplate(self, b):
        if b.shape != [3, 3, 1, 1]:
            b = np.reshape(b, [3, 3, 1, 1])
        self.B.assign(b)

    def Simulate(self, toNumpy=False):
        val = self.CellEquation()
        if toNumpy:
            return val.numpy()
        return val

    def CellEquation(self):
        padded_input = self.Input
        padded_state = self.State

        if self.Boundary == BoundaryTypes.CONSTANT:
            padded_input = tf.pad(
                padded_input,
                [[0, 0], [1, 1], [1, 1], [0, 0]],
                "CONSTANT",
                constant_values=self.BoundValue
            )

            padded_state = tf.pad(
                padded_state,
                [[0, 0], [1, 1], [1, 1], [0, 0]],
                "CONSTANT",
                constant_values=self.BoundValue
            )
        elif self.Boundary == BoundaryTypes.ZERO_FLUX:
            padded_input = tf.pad(
                padded_input,
                [[0, 0], [1, 1], [1, 1], [0, 0]],
                "SYMMETRIC"
            )

            padded_state = tf.pad(
                padded_state,
                [[0, 0], [1, 1], [1, 1], [0, 0]],
                "SYMMETRIC"
            )
        elif self.Boundary == BoundaryTypes.PERIODIC:
            padded_input = tf.pad(
                padded_input,
                [[0, 0], [1, 1], [1, 1], [0, 0]],
                "REFLECT"
            )

            padded_state = tf.pad(
                padded_state,
                [[0, 0], [1, 1], [1, 1], [0, 0]],
                "REFLECT"
            )

        BU = tf.nn.conv2d(padded_input, self.B, strides=[1, 1, 1, 1], padding='VALID')
        BU = BU + self.Z

        x = padded_state
        iterations = int(self.SimTime / self.TimeStep)
        for it in range(iterations):
            y = tf.maximum(tf.minimum(x, 1), -1)
            AY = tf.nn.conv2d(y, self.A, strides=[1, 1, 1, 1], padding='VALID')  # VALID, SAME
            x = x + self.TimeStep * (-1 * x + BU + AY)
        out = tf.maximum(tf.minimum(x, 1), -1)
        return out

    def SaveNetwork(self, filename):
        A = self.A.numpy()
        B = self.B.numpy()
        Z = self.Z.numpy()

        np.savez(filename, TemplateA=A, TemplateB=B, bias=Z, score=self.ModelBestLoss)

    def LoadNetwork(self, file):
        if not exists(file):
            print(file + " cannot be opened, skipping...")
            return
        data = np.load(file)
        a = data["TemplateA"]
        b = data["TemplateB"]
        z = data["bias"][0]
        l = data["score"]

        self.SetA(a)
        self.SetB(b)
        self.SetZ(z)
        self.ModelBestLoss = l


@tf.function
def ComputeLoss(expectedOutput, modelOutput):
    return tf.reduce_mean(tf.abs(expectedOutput - modelOutput))


def TrainingWrapper():
    @tf.function
    def TrainingStep(model: CellularNetwork, exp_out):
        with tf.GradientTape() as tape:
            model_out = model.Simulate()
            loss = ComputeLoss(exp_out, model_out)

        grads = tape.gradient(loss, [model.A, model.B, model.Z])
        model.Optimizer.apply_gradients(zip(grads, [model.A, model.B, model.Z]))
        return loss

    return TrainingStep
