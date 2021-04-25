'''from enum import Enum
import tensorflow as tf
import cv2
from cnn import ImageToCell
import numpy as np

class BoundaryTypes(Enum):
    CONSTANT = 0,
    ZERO_FLUX = 1,
    PERIODIC = 2

class MockCNN:
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
        else:
            tempImg = In
        tempImg = np.reshape(tempImg, [1, tempImg.shape[0], tempImg.shape[1], 1])
        self.Input = tf.constant(tempImg, dtype=tf.float32)

    def SetState(self, St):
        if isinstance(St, str):
            tempImg = ImageToCell(cv2.cvtColor(cv2.imread(St), cv2.COLOR_BGR2GRAY))
        else:
            tempImg = St
        tempImg = np.reshape(tempImg, [1, tempImg.shape[0], tempImg.shape[1], 1])
        self.State = tf.constant(tempImg, dtype=tf.float32)

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

    def Simulate(self):
        return self.CellEquation()

    def CellEquation(self):
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
def compute_loss(exp_out, modelout):
    return tf.reduce_mean(tf.abs(exp_out - modelout))


@tf.function
def TrainingStep(model: MockCNN, exp_out):
    with tf.GradientTape() as tape:
        model_out = model.Simulate()
        loss = compute_loss(exp_out, model_out)

    grads = tape.gradient(loss, [model.A, model.B, model.Z])
    model.Optimizer.apply_gradients(zip(grads, [model.A, model.B, model.Z]))
    return loss

### TESTING STARTS HERE

img = cv2.imread('images/avergra2.png')
img = ImageToCell(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

bimg = np.reshape(img, [1, img.shape[0], img.shape[1], 1])

x0 = tf.constant(bimg, dtype=tf.float32)
U = tf.constant(bimg, dtype=tf.float32)

mcnn = MockCNN()
mcnn.SetInput("images/avergra2.png")
mcnn.SetState("images/avergra2.png")

expout = np.load('ExpectedOutput.npy')
expected_output = np.reshape(expout, [1, img.shape[0], img.shape[1], 1])
optimizer = tf.optimizers.Adam(0.1)

for i in range(1000):
    loss = TrainingStep(mcnn, expected_output)
    if i % 100 == 0:
        print(loss)
'''