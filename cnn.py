import numpy as np
import cv2


def ImageToCell(image):
    image = (-1) * ((image.astype(np.float)) / 128.0 - 1.0)
    return image


def CellToImage(cell):
    cell = ((cell * -1) + 1.0) * 127.5
    return cell.astype(np.uint8)


def StandardCNNNonliearity(x):
    # this function implements the standard CNN nonlinearity, all values are saturated below -1 and above 1
    back = x
    back[x < -1] = -1
    back[x > 1] = 1
    return back


class CellSim():
    def __init__(self):
        self.Input = []
        self.State = []
        self.A = np.zeros((3, 3))
        self.B = np.zeros((3, 3))
        self.Z = 0
        self.SimTime = 1
        self.TimeStep = 0.1
        self.OutputNonlin = StandardCNNNonliearity
        self.Boundary = 'Constant'
        self.BoundValue = 0

    def GetOutput(self):
        return self.Output

    def SetTimeStep(self, Ts):
        # this function sets the A template of the simulator
        # check if it is an N times N matrix - later on these could be functions
        self.TimeStep = Ts

    def SetSimTime(self, T):
        # this function sets the A template of the simulator
        # check if it is an N times N matrix - later on these could be functions
        self.SimTime = T

    def SetInput(self, In):
        # this function sets the A template of the simulator
        # check if it is an N times N matrix - later on these could be functions
        # and convert image to CellNN domain
        if isinstance(In, str):
            img = ImageToCell(cv2.cvtColor(cv2.imread(In), cv2.COLOR_BGR2GRAY))
        else:
            img = In
        self.Input = img

    def SetState(self, St):
        # this function sets the A template of the simulator
        # check if it is an N times N matrix - later on these could be functions
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
        # this function sets the A template of the simulator
        # check if it is an N times N matrix - later on these could be functions
        self.A = a

    def SetB(self, a):
        self.SetBTemplate(a)

    def SetBTemplate(self, b):
        # this function sets the A template of the simulator
        # check if it is an N times N matrix - later on these could be functions
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

        # r = ode(self.cell_equation).set_integrator('vode', method='bdf', with_jacobian=False)
        # r = ode(self.cell_equation).set_integrator('dopri')
        # r.set_initial_value(self.State.flatten(), 0)
        # start= time.time()
        # while r.successful() and r.t < self.SimTime:
        #   r.integrate(r.t+self.TimeStep)
        #   print(r.t)
        # end= time.time()
        # print(end-start)
        # SizeX=self.State.shape[0]
        # SizeY=self.State.shape[1]
        # OutImg=self.OutputNonlin(np.reshape(r.y,[SizeX,SizeY]))
        return OutImg

    def cell_equation(self, t, X):
        # This function impelment the differential equation determining the standard cnn cell:
        # xdot = -x + Ay + Bu + z
        # the parameters of the CNN array (templates) are stored in P

        # reshape the 1xN input for the size of the image -ode solvers can only deal with vectors but code is more understandable if    we use arrays
        SizeX = self.State.shape[0]
        SizeY = self.State.shape[1]
        x = np.reshape(X, [SizeX, SizeY])

        # we will return the derivative in this array
        dx = np.zeros((SizeX, SizeY))

        # go through all elements of the array
        for a in range(SizeX):
            for b in range(SizeY):
                # if we are at the edge of the array, boundary conditions should be applied
                if (a == 0) or (b == 0) or (a == (SizeX - 1)) or (b == (SizeY - 1)):
                    inputregion = np.zeros((3, 3))
                    stateregion = np.zeros((3, 3))
                    # check the local region around the cell
                    for c in range(-1, 2):
                        for d in range(-1, 2):
                            # check boundary conditions if we are at the edge of the array
                            if (self.Boundary == 'Constant'):
                                # constant boundary condition, virtual cells have fix values
                                if (a + c < 0) | (b + d < 0) | (a + c > (SizeX - 1)) | (b + d > (SizeY - 1)):
                                    inputregion[c + 1, d + 1] = self.BoundValue
                                    stateregion[c + 1, d + 1] = self.BoundValue
                                else:
                                    inputregion[c + 1, d + 1] = self.Input[a + c, b + d]
                                    stateregion[c + 1, d + 1] = x[a + c, b + d]
                            elif self.Boundary == 'ZeroFlux':
                                # zero-flux condition- virtual cells have the value of the closes real cell
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
                                # periodic condition- the value of the next real cell at the other edge of the array will be used
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
                                inputregion[c + 1, d + 1] = P.Input[inda, indb]
                                stateregion[c + 1, d + 1] = x[inda, indb]

                else:
                    # if we are not at the edge, just select the region, all cells are valid
                    inputregion = self.Input[a - 1:a + 2, b - 1:b + 2]
                    stateregion = x[a - 1:a + 2, b - 1:b + 2]

                y = self.OutputNonlin(stateregion)
                # calculate the derivative according to the equation
                dx[a, b] = -x[a, b] + np.sum(np.multiply(self.A, y)) + np.sum(np.multiply(self.B, inputregion)) + self.Z
        # reshape back to Nx1
        dx = np.reshape(dx, [SizeX * SizeY])

        return dx