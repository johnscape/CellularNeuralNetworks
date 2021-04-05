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


def test_standard_cnnnonlinearity():
    inp = np.zeros((3, 3))
    exp = np.zeros((3, 3))

    inp[0, 0] = -1
    inp[0, 1] = 1
    inp[0, 2] = 5

    exp[0, 0] = -1
    exp[0, 1] = 1
    exp[0, 2] = 1

    inp[1, 0] = 0
    inp[1, 1] = 0.2
    inp[1, 2] = -0.2

    exp[1, 0] = 0
    exp[1, 1] = 0.2
    exp[1, 2] = -0.2

    inp[2, 0] = -5
    inp[2, 1] = 2.2
    inp[2, 2] = -1.1

    exp[2, 0] = -1
    exp[2, 1] = 1
    exp[2, 2] = -1

    ans = StandardCNNNonlinearity(inp)

    assert np.array_equal(ans, exp) == True


def test_find_active_regions_constant():
    cnn = CellularNetwork()

    cnn.Boundary = BoundaryTypes.CONSTANT
    cnn.BoundValue = 1

    inp = np.ones((5, 5)) * 2
    state = np.ones((5, 5)) * 5

    cnn.SetInput(inp)
    cnn.SetState(state)

    expected_input_area = np.asarray([
        [1, 1, 1],
        [1, 2, 2],
        [1, 2, 2]
    ])

    expected_state_area = np.asarray([
        [1, 1, 1],
        [1, 5, 5],
        [1, 5, 5]
    ])

    input_area, state_area = cnn.FindActiveRegions(0, 0, state)

    assert np.array_equal(input_area, expected_input_area) and \
           np.array_equal(state_area, expected_state_area)


def test_find_active_regions_zero():
    cnn = CellularNetwork()

    cnn.Boundary = BoundaryTypes.ZERO_FLUX
    cnn.BoundValue = 1

    inp = np.ones((5, 5))
    state = np.ones((5, 5))

    for x in range(5):
        for y in range(5):
            inp[x, y] = (x + 1) * (y + 1)
            state[x, y] = (x + 1) * (y + 1) + 2

    cnn.SetInput(inp)
    cnn.SetState(state)

    expected_input_area = np.asarray([
        [1, 1, 2],
        [1, 1, 2],
        [2, 2, 4]
    ])

    expected_state_area = np.asarray([
        [3, 3, 4],
        [3, 3, 4],
        [4, 4, 6]
    ])

    input_area, state_area = cnn.FindActiveRegions(0, 0, state)

    assert np.array_equal(input_area, expected_input_area) and \
           np.array_equal(state_area, expected_state_area)


def test_find_active_regions_periodic():
    cnn = CellularNetwork()

    cnn.Boundary = BoundaryTypes.PERIODIC
    cnn.BoundValue = 1

    inp = np.ones((5, 5))
    state = np.ones((5, 5))

    for x in range(5):
        for y in range(5):
            inp[x, y] = (x + 1) * (y + 1)
            state[x, y] = (x + 1) * (y + 1) + 2

    cnn.SetInput(inp)
    cnn.SetState(state)

    expected_input_area = np.asarray([
        [25, 5, 10],
        [5, 1, 2],
        [10, 2, 4]
    ])

    expected_state_area = np.asarray([
        [27, 7, 12],
        [7, 3, 4],
        [12, 4, 6]
    ])

    input_area, state_area = cnn.FindActiveRegions(0, 0, state)

    assert np.array_equal(input_area, expected_input_area) and \
           np.array_equal(state_area, expected_state_area)
