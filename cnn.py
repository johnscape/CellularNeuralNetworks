import numpy as np
import cv2
from enum import Enum


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
    back = x
    back[x < -1] = -1
    back[x > 1] = 1
    return back


class CellularNetwork:
    def __init__(self):
        self.Input = []
        self.State = []
        self.A = np.zeros((3, 3))
        self.B = np.zeros((3, 3))
        self.Z = 0
        self.SimTime = 1
        self.TimeStep = 0.1
        self.OutputNonlin = StandardCNNNonlinearity
        self.Boundary = BoundaryTypes.CONSTANT
        self.BoundValue = 0

    def GetOutput(self):
        return self.Output

    def SetTimeStep(self, Ts):
        self.TimeStep = Ts

    def SetSimTime(self, T):
        self.SimTime = T

    def SetInput(self, In):
        if isinstance(In, str):
            img = ImageToCell(cv2.cvtColor(cv2.imread(In), cv2.COLOR_BGR2GRAY))
        else:
            img = In
        self.Input = img

    def SetState(self, St):
        if isinstance(St, str):
            img = ImageToCell(cv2.cvtColor(cv2.imread(St), cv2.COLOR_BGR2GRAY))
        else:
            img = St
        self.State = img

    def SetZ(self, z):
        self.SetBias(z)

    def SetBias(self, z):
        self.Z = z

    def SetA(self, a):
        self.SetATemplate(a)

    def SetATemplate(self, a):
        self.A = a

    def SetB(self, a):
        self.SetBTemplate(a)

    def SetBTemplate(self, b):
        self.B = b

    def Euler(self, f, y0, StartTime, EndTime, h):
        t, y = StartTime, y0
        while t <= EndTime:
            t += h
            y += h * f(t, y)
        return y

    def Simulate(self):
        self.Input = self.Input.astype(np.float64)
        self.State = self.State.astype(np.float64)
        Ret = self.Euler(self.cell_equation, self.State.flatten(), 0, self.SimTime, 0.1)
        SizeX = self.State.shape[0]
        SizeY = self.State.shape[1]
        OutImg = self.OutputNonlin(np.reshape(Ret, [SizeX, SizeY]))

        return OutImg

    def cell_equation(self, t, X):
        SizeX = self.State.shape[0]
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

        return dx

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
