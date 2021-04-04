from cnn import CellularNetwork, BoundaryTypes, CellToImage
import cv2


# TASK 1
def DoTaskOne(saveImg=False):
    threshold = 0.6

    cnn = CellularNetwork()
    cnn.SetInput("images/task_1.bmp")
    cnn.SetState("images/task_1.bmp")

    cnn.SetTimestep(0.1)
    cnn.SetMaxTime(10)

    cnn.Boundary = BoundaryTypes.ZERO_FLUX

    cnn.SetA([[0, 0, 0], [0, 2.0, 0], [0, 0, 0]])
    cnn.SetB([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    cnn.SetBias(-((2 * 0.6) - 1))  # -0.2

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

    cnn.SetTimestep(0.1)
    cnn.SetMaxTime(10)

    cnn.Boundary = BoundaryTypes.ZERO_FLUX

    cnn.SetA([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    cnn.SetB([[0, 1.0, 0], [0, 1.0, 1.0], [0, 0, 0]])
    cnn.SetBias(-2.0)  # -0.2

    print("Starting simulation for Task 2")
    img = cnn.Simulate()
    print("Simulation finished! Showing image...")
    cvImg = CellToImage(img)
    cv2.imshow("Erosion", cvImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if saveImg:
        cv2.imwrite("task_2_out.bmp", cvImg)


#DoTaskOne()
DoTaskTwo()