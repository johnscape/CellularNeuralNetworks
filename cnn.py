import numpy as np
import cv2
from enum import Enum
import tensorflow as tf


class BoundaryTypes(Enum):
    CONSTANT = 0,
    ZERO_FLUX = 1,
    PERIODIC = 2


def ComputeLoss(expected, output):
    return tf.reduce_mean(tf.abs(expected - output))


def TrainingStep(model, optimizer, expected):
    with tf.GradientTape() as tape:
        model_out = model.CellEquation(False)
        loss = ComputeLoss(expected, model_out)

        gradients = tape.gradient(loss, [model.A, model.B, model.Z])
        optimizer.apply_gradients(zip(gradients, [model.A, model.B, model.Z]))

        return loss


def ImageToCell(image):
    image = (-1) * ((image.astype(np.float)) / 127.5 - 1.0)
    return image


def CellToImage(cell):
    cell = ((cell * -1) + 1.0) * 127.5
    return cell.astype(np.uint8)


def StandardCNNNonlinearity(x):
    return tf.clip_by_value(x, -1, 1)


def TensorToNumpy(tensor):
    init_g = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_g)
        sess.run(init_l)
        arr = sess.run(tensor)
        return arr.astype(np.float)


class CellularNetwork:
    def __init__(self):
        self.Input = []
        self.State = []

        initializer = tf.initializers.glorot_uniform()

        self.A = tf.Variable(initializer([3, 3, 1, 1]), dtype=tf.float32)
        self.B = tf.Variable(initializer([3, 3, 1, 1]), dtype=tf.float32)
        self.Z = tf.Variable(initializer([1]), dtype=tf.float32)
        self.SimTime = 1
        self.TimeStep = 0.1
        self.OutputNonlin = StandardCNNNonlinearity
        self.Boundary = BoundaryTypes.CONSTANT
        self.BoundValue = 0

    def GetOutput(self):
        return self.State

    def SetTimeStep(self, Ts):
        self.TimeStep = Ts

    def SetSimTime(self, T):
        self.SimTime = T

    def SetInput(self, In):
        if isinstance(In, str):
            img = ImageToCell(cv2.cvtColor(cv2.imread(In), cv2.COLOR_BGR2GRAY))
        else:
            img = In
        self.Input = tf.constant(img, dtype=tf.float32)
        self.Input = tf.reshape(self.Input, [1, self.Input.shape[0], self.Input.shape[1], 1])

    def SetState(self, St):
        if isinstance(St, str):
            img = ImageToCell(cv2.cvtColor(cv2.imread(St), cv2.COLOR_BGR2GRAY))
        else:
            img = St
        self.State = tf.Variable(img, dtype=tf.float32)
        self.State = tf.reshape(self.State, [1, self.State.shape[0], self.State.shape[1], 1])

    def SetZ(self, z):
        self.SetBias(z)

    def SetBias(self, z):
        self.Z = tf.constant(z, dtype=tf.float32)

    def SetA(self, a):
        self.SetATemplate(a)

    def SetATemplate(self, a):
        self.A = tf.reshape(tf.Variable(a, dtype=tf.float32), [3, 3, 1, 1])

    def SetB(self, a):
        self.SetBTemplate(a)

    def SetBTemplate(self, b):
        self.B = tf.reshape(tf.Variable(b, dtype=tf.float32), [3, 3, 1, 1])

    def Simulate(self, toNumpy=True):
        Ret = self.OutputNonlin(self.CellEquation())
        if toNumpy:
            return TensorToNumpy(Ret[0, :, :, 0])
        else:
            return Ret

    def CellEquation(self):  # TODO: implement different padding methods
        BU = tf.nn.conv2d(self.Input, self.B, strides=[1, 1, 1, 1], padding='SAME')
        BU += self.Z

        time_step = tf.constant(self.TimeStep, dtype=tf.float32)

        current_time = 0
        while current_time < self.SimTime:
            self.State = self.OutputNonlin(self.State)
            AY = tf.nn.conv2d(self.State, self.A, strides=[1, 1, 1, 1], padding='SAME')
            dx = -self.State + AY + BU

            self.State += dx * time_step
            current_time += self.TimeStep
        return self.State
