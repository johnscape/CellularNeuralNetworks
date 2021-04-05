from cnn import CellularNetwork, BoundaryTypes, CellToImage
import cv2
import numpy as np

# TASK 1
def DoTaskOne(saveImg=False):
    threshold = 0.6

    cnn = CellularNetwork()
    cnn.SetInput("images/task_1.bmp")
    cnn.SetState("images/task_1.bmp")

    cnn.SetTimeStep(0.1)
    cnn.SetSimTime(10)

    cnn.Boundary = BoundaryTypes.CONSTANT
    cnn.ConstantBoundary = 0

    cnn.SetA([[0, 0, 0], [0, 2.0, 0], [0, 0, 0]])
    cnn.SetB([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    cnn.SetBias(-((2 * threshold) - 1))  # -0.2

    print("Starting simulation for Task 1")
    img = cnn.Simulate()
    print("Simulation finished! Showing image...")
    cvImg = CellToImage(img)
    cv2.imshow("Threshold", cvImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if saveImg:
        cv2.imwrite("task_1_out.bmp", cvImg)


def DoTaskTwo(saveImg=False):
    cnn = CellularNetwork()
    cnn.SetInput("images/task_2.bmp")
    cnn.SetState("images/task_2.bmp")

    cnn.SetTimeStep(0.1)
    cnn.SetSimTime(10)

    cnn.Boundary = BoundaryTypes.CONSTANT
    cnn.ConstantBoundary = 0

    cnn.SetA([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    cnn.SetB([[0, 1.0, 0], [0, 1.0, 1.0], [0, 0, 0]])
    cnn.SetBias(-2.0)

    print("Starting simulation for Task 2")
    img = cnn.Simulate()
    print("Simulation finished! Showing image...")
    cvImg = CellToImage(img)
    cv2.imshow("Erosion", cvImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if saveImg:
        cv2.imwrite("task_2_out.bmp", cvImg)


def DoTaskThree(saveImg=False):
    cnn = CellularNetwork()
    cnn.SetInput("images/task_3.bmp")
    cnn.SetState(np.ones((100, 100)))

    cnn.SetTimeStep(1)
    cnn.SetSimTime(100)

    cnn.Boundary = BoundaryTypes.CONSTANT
    cnn.ConstantBoundary = 0

    cnn.SetA([[0, 0, 0], [0, 2.0, 0], [0, 2.0, 0]])
    cnn.SetB([[0, 0, 0], [0, 2.0, 0], [0, 0, 0]])
    cnn.SetBias(0)

    print("Starting simulation for Task 3")
    img = cnn.Simulate()
    print("Simulation finished! Showing image...")
    cvImg = CellToImage(img)
    cv2.imshow("ShadowUp", cvImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if saveImg:
        cv2.imwrite("task_3_out.bmp", cvImg)


# DoTaskOne()
# DoTaskTwo(True)
#DoTaskThree(True)
