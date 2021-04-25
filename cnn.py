import numpy as np
import cv2
from enum import Enum
import tensorflow as tf


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

    def GetOutput(self):
        return self.State

    def SetTimeStep(self, Ts):
        self.TimeStep = Ts

    def SetSimTime(self, T):
        self.SimTime = T

    def SetInput(self, In):
        if isinstance(In, str):
            tempImg = ImageToCell(cv2.cvtColor(cv2.imread(In), cv2.COLOR_BGR2GRAY))
            tempImg = np.reshape(tempImg, [1, tempImg.shape[0], tempImg.shape[1], 1])
        else:
            tempImg = In

        if self.Input is None:
            self.Input = tf.Variable(tempImg, dtype=tf.float32)
        else:
            self.Input.assign(tempImg, dtype=tf.float32)

    def SetState(self, St):
        if isinstance(St, str):
            tempImg = ImageToCell(cv2.cvtColor(cv2.imread(St), cv2.COLOR_BGR2GRAY))
            tempImg = np.reshape(tempImg, [1, tempImg.shape[0], tempImg.shape[1], 1])
        else:
            tempImg = St
        if self.State is None:
            self.State = tf.Variable(tempImg, dtype=tf.float32)
        else:
            self.State.assign(tempImg, dtype=tf.float32)

    def SetZ(self, z):
        self.SetBias(z)

    def SetBias(self, z):
        self.Z = tf.Variable(z, dtype=tf.float32)

    def SetA(self, a):
        self.SetATemplate(a)

    def SetATemplate(self, a):
        self.A = tf.reshape(tf.Variable(a, dtype=tf.float32), [3, 3, 1, 1])

    def SetB(self, a):
        self.SetBTemplate(a)

    def SetBTemplate(self, b):
        self.B = tf.reshape(tf.Variable(b, dtype=tf.float32), [3, 3, 1, 1])

    def Simulate(self, toNumpy=True):
        return self.CellEquation()

    def CellEquation(self):  # TODO: implement different padding methods
        BU = tf.nn.conv2d(self.Input, self.B, strides=[1, 1, 1, 1], padding='SAME')
        BU = BU + self.Z

        x = self.State
        iterations = int(self.SimTime / self.TimeStep)
        for it in range(iterations):
            y = tf.maximum(tf.minimum(x, 1), -1)
            AY = tf.nn.conv2d(y, self.A, strides=[1, 1, 1, 1], padding='SAME')  # VALID, SAME
            x = x + self.TimeStep * (-1 * (x) + BU + AY)
        out = tf.maximum(tf.minimum(x, 1), -1)
        return out


@tf.function
def ComputeLoss(expectedOutput, modelOutput):
    return tf.reduce_mean(tf.abs(expectedOutput - modelOutput))


@tf.function
def TrainingStep(model: CellularNetwork, exp_out):
    with tf.GradientTape() as tape:
        model_out = model.Simulate()
        loss = ComputeLoss(exp_out, model_out)

    grads = tape.gradient(loss, [model.A, model.B, model.Z])
    model.Optimizer.apply_gradients(zip(grads, [model.A, model.B, model.Z]))
    return loss
