import numpy as np
import cv2
from enum import Enum


class BoundaryTypes(Enum):
    CONSTANT = 0,
    ZERO_FLUX = 1,
    PERIODIC = 2


def ImageToCell(image):
    image = (-1) * ((image.astype(np.float)) / 128.0 - 1.0)
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
                if (a == 0) or (b == 0) or (a == (SizeX - 1)) or (b == (SizeY - 1)):
                    inputregion = np.zeros((3, 3))
                    stateregion = np.zeros((3, 3))
                    for c in range(-1, 2):
                        for d in range(-1, 2):
                            if (self.Boundary == 'Constant'):
                                if (a + c < 0) | (b + d < 0) | (a + c > (SizeX - 1)) | (b + d > (SizeY - 1)):
                                    inputregion[c + 1, d + 1] = self.BoundValue
                                    stateregion[c + 1, d + 1] = self.BoundValue
                                else:
                                    inputregion[c + 1, d + 1] = self.Input[a + c, b + d]
                                    stateregion[c + 1, d + 1] = x[a + c, b + d]
                            elif self.Boundary == 'ZeroFlux':
                                inda = a + c
                                if a + c < 0:
                                    inda = 0
                                elif a + c > (SizeX - 1):
                                    inda = SizeX - 1
                                indb = b + d
                                if b + d < 0:
                                    indb = 0
                                elif b + d > (SizeY - 1):
                                    indb = SizeY - 1
                                inputregion[c + 1, d + 1] = self.Input[inda, indb]
                                stateregion[c + 1, d + 1] = x[inda, indb]
                            elif self.Boundary == 'Periodic':
                                inda = a + c
                                if a + c < 0:
                                    inda = SizeX - 1
                                elif a + c > SizeX - 1:
                                    inda = 0
                                indb = b + d
                                if b + d < 0:
                                    indb = SizeY - 1
                                elif b + d > SizeY - 1:
                                    indb = 0
                                inputregion[c + 1, d + 1] = self.Input[inda, indb]
                                stateregion[c + 1, d + 1] = x[inda, indb]

                else:
                    inputregion = self.Input[a - 1:a + 2, b - 1:b + 2]
                    stateregion = x[a - 1:a + 2, b - 1:b + 2]

                y = self.OutputNonlin(stateregion)
                dx[a, b] = -x[a, b] + np.sum(np.multiply(self.A, y)) + np.sum(np.multiply(self.B, inputregion)) + self.Z
        dx = np.reshape(dx, [SizeX * SizeY])

        return dx
