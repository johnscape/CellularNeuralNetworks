import numpy as np
import cv2
import copy
from enum import Enum
from collections.abc import Callable


def ImageToCell(image: np.array) -> np.array:  # 0-255 -> -1 - 1
    cellData = copy.deepcopy(image)
    cellData = (cellData / 255) * 2 - 1
    cellData *= -1
    return cellData.astype(int)


def CellToImage(cell: np.array) -> np.array:
    image = copy.deepcopy(cell)
    image *= -1
    image = (image + 1) / 2 * 255
    return image.astype(int)


def Euler(function, y0, startTime, endTime, timeStep) -> np.array:
    t, y = startTime, y0
    while t <= endTime:
        t += timeStep
        y += timeStep * function(t, y)
    return y


def StandardCNNNonliearity(x: float) -> float:
    if x < -1:
        return -1
    elif x > 1:
        return 1
    return x


class BoundaryTypes(Enum):
    ZERO_FLUX = 0,
    CONSTANT = 1,
    PERIODIC = 2


class CellularNetwork:
    def __init__(self):
        self.Input = []
        self.State = []
        self.Output = []
        self.A = np.zeros((3, 3))
        self.B = np.zeros((3, 3))
        self.Z = 0
        self.SimTime = 1
        self.TimeStep = 0.1
        self.NonlinearFunction = StandardCNNNonliearity
        self.Boundary = BoundaryTypes.ZERO_FLUX
        self.ConstantBoundaty = 0

    def GetOutput(self):
        return self.Output

    def SetTimestep(self, timeStep: float):
        self.TimeStep = timeStep

    def SetInput(self, fileName: str):
        readImg = cv2.imread(fileName)
        readImg = cv2.cvtColor(readImg, cv2.COLOR_BGR2GRAY)
        self.Input = ImageToCell(readImg)

    def SetState(self, fileName: str):
        readImg = cv2.imread(fileName)
        readImg = cv2.cvtColor(readImg, cv2.COLOR_BGR2GRAY)
        self.State = ImageToCell(readImg)

    def SetBias(self, bias: float):
        self.Z = bias

    def SetA(self, a: np.array):
        self.A = a

    def SetB(self, b: np.array):
        self.B = b

    def Simulate(self):
        r = Euler(self.CellFunction, self.State.flatten(), 0, self.SimTime, 0.1)
        x = self.State.shape[0]
        y = self.State.shape[1]

        image = self.NonlinearFunction(np.reshape(r, [x, y]))
        return image

    def CellFunction(self, t: float, X: np.array) -> np.array:
        x = self.State.shape[0]
        y = self.State.shape[1]

        input_data = np.reshape(X, [x, y])
        dx = np.zeros((x, y))

        self.Output = self.NonlinearFunction(input_data)

        for a in range(x):
            for b in range(y):
                active_input_area = np.zeros((3, 3))
                active_output_area = np.zeros((3, 3))
                if a == 0 or b == 0 or a == x - 1 or b == y - 1:
                    for t_x in range(-1, 2):
                        for t_y in range(-1, 2):
                            active_output_area[t_x + 1, t_y + 1] = self.GetBoundaryValue(a + t_x, b + t_y, True)
                            active_input_area[t_x + 1, t_y + 1] = self.GetBoundaryValue(a + t_x, b + t_y)
                else:
                    active_input_area = self.Input[(a - 1):(a + 2), (b - 1):(b + 2)]
                    active_output_area = self.Output[(a - 1):(a + 2), (b - 1):(b + 2)]
                BU = np.sum(np.multiply(self.B, active_input_area))
                AY = np.sum(np.multiply(self.A, active_output_area))
                dx[a, b] = -input_data[a, b] + AY + BU + self.Z

        dx = np.reshape(dx, [x * y])

        return dx

    def GetBoundaryValue(self, x, y, output=False):
        if x < 0 or y < 0 or x >= self.State.shape[0] or y >= self.State.shape[1]:
            if self.Boundary == BoundaryTypes.ZERO_FLUX:
                return 0
            elif self.Boundary == BoundaryTypes.CONSTANT:
                return self.ConstantBoundaty
            elif self.Boundary == BoundaryTypes.PERIODIC:
                if x < 0: x += self.State.shape[0]
                elif x >= self.State.shape[0]: x -= self.State.shape[0]

                if y < 0: y += self.State.shape[1]
                elif y >= self.State.shape[1]: y -= self.State.shape[1]

                if output:
                    return self.Output[x, y]
                return self.Input[x, y]
        else:
            if output:
                return self.Output[x, y]
            return self.Input[x, y]
