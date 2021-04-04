import pytest
import numpy as np
from cnn import *

imgToCellData = [
    (np.zeros((15, 15)), np.ones((15, 15))),
    (255 * np.ones((15, 15)), -1 * np.ones((15, 15)))
]

cellToImgData = [
    (np.ones((15, 15)), np.zeros((15, 15))),
    (-1 * np.ones((15, 15)), 255 * np.ones((15, 15)))
]


@pytest.mark.parametrize("inp, expected", imgToCellData)
def test_image_to_cell(inp, expected):
    r = ImageToCell(inp)
    assert np.array_equal(r, expected) is True, "Error was " + str(ImageToCell(inp)[0])


@pytest.mark.parametrize("inp, expected", cellToImgData)
def test_cell_to_image(inp, expected):
    r = CellToImage(inp)
    assert np.array_equal(r, expected) is True


def test_standard_cnnnonliearity():
    assert StandardCNNNonliearity(15) == 1


def test_get_boundary_value():
    CNN = CellularNetwork()
    CNN.Boundary = BoundaryTypes.PERIODIC
    CNN.Input = np.zeros((5, 5))
    CNN.Output = np.zeros((5, 5))
    CNN.State = np.zeros((5, 5))

    for x in range(5):
        for y in range(5):
            CNN.Input[x, y] = (x + 1) * (y + 1)

    expected = np.asarray([
        [25, 5, 10],
        [5, 1, 2],
        [10, 2, 4]
    ])

    b = np.zeros((3, 3))
    s_x = 0
    s_y = 0
    for x in range(-1, 2):
        for y in range(-1, 2):
            b[x + 1, y + 1] = CNN.GetBoundaryValue(s_x + x, s_y + y)

    assert np.array_equal(b, expected) is True
