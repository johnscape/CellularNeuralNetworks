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

    def Euler(self, f, y0, StartTime, EndTime, h):
        t, y = StartTime, y0
        while t <= EndTime:
            t += h
            y += h * f(t, y)
        return y

    def Simulate(self):
        # self.Input = self.Input.astype(np.float64)
        # self.State = self.State.astype(np.float64)
        Ret = self.CellEquation(self.SimTime / self.TimeStep)
        Ret = self.OutputNonlin(Ret)
        return TensorToNumpy(Ret[0, :, :, 0])

    def CellEquation(self, t):
        '''SizeX = self.State.shape[0]
        SizeY = self.State.shape[1]
        x = np.reshape(X, [SizeX, SizeY])

        dx = np.zeros((SizeX, SizeY))

        for a in range(SizeX):
            for b in range(SizeY):
                input_region, state_region = self.FindActiveRegions(a, b, x)

                y = self.OutputNonlin(state_region)
                AY = np.sum(np.multiply(self.A, y))
                BU = np.sum(np.multiply(self.B, input_region))
                dx[a, b] = -x[a, b] + AY + BU + self.Z
        dx = np.reshape(dx, [SizeX * SizeY])

        return dx'''

        '''AY = tf.nn.conv2d(y, self.A, strides=[1, 1, 1, 1], padding="SAME")
        BU = tf.nn.conv2d(self.Input, self.B, strides=[1, 1, 1, 1], padding="SAME")

        dx = -y + AY + BU + self.Z
        return dx'''

        BU = tf.nn.conv2d(self.Input, self.B, strides=[1, 1, 1, 1], padding='SAME')
        BU += self.Z

        time_step = tf.constant(self.TimeStep, dtype=tf.float32)

        for time in range(int(t)):
            self.State = self.OutputNonlin(self.State)
            AY = tf.nn.conv2d(self.State, self.A, strides=[1, 1, 1, 1], padding='SAME')
            dx = -self.State + AY + BU

            self.State += dx * time_step
        return self.State

    def FindActiveRegions(self, x, y, currentData):
        input_region = np.zeros((3, 3))
        state_region = np.zeros((3, 3))
        if 0 < x < self.State.shape[0] - 1 and 0 < y < self.State.shape[1] - 1:
            input_region = self.Input[x - 1:x + 2, y - 1:y + 2]
            state_region = currentData[x - 1:x + 2, y - 1:y + 2]
        else:
            for i in range(-1, 2):
                for ii in range(-1, 2):
                    input_region[i + 1, ii + 1] = self.FindRealValues(x + i, y + ii, None)
                    state_region[i + 1, ii + 1] = self.FindRealValues(x + i, y + ii, currentData)

        return input_region, state_region

    def FindRealValues(self, x, y, state):
        if 0 <= x < self.State.shape[0] and 0 <= y < self.State.shape[1]:
            if state is not None:
                return state[x, y]
            return self.Input[x, y]
        if self.Boundary == BoundaryTypes.CONSTANT:
            return self.BoundValue
        if self.Boundary == BoundaryTypes.ZERO_FLUX:
            if x < 0:
                x = 0
            elif x > self.State.shape[0] - 1:
                x = self.State.shape[0] - 1

            if y < 0:
                y = 0
            elif y > self.State.shape[1] - 1:
                y = self.State.shape[1] - 1

            if state is not None:
                return state[x, y]
            return self.Input[x, y]
        if self.Boundary == BoundaryTypes.PERIODIC:
            if x < 0:
                x += self.State.shape[0]
            elif x > self.State.shape[0] - 1:
                x -= self.State.shape[0]

            if y < 0:
                y += self.State.shape[1]
            elif y > self.State.shape[1] - 1:
                y -= self.State.shape[1]

            if state is not None:
                return state[x, y]
            return self.Input[x, y]

        raise Exception("Unknown location or boundary method")
