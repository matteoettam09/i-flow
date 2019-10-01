""" Test spline implementation. """

import pytest

import numpy as np

from flow.splines import spline
from flow.splines import linear_spline
from flow.splines import quadratic_spline
from flow.splines import cubic_spline
from flow.splines import rational_quadratic_spline


def test_spline_utilities():
    """ Test the utility functions in spline. """
    array = np.ones(3)
    array = spline._padded(array, 0, 0)  # pylint: disable=protected-access
    assert np.all(np.equal(array, np.array([0, 1, 1, 1, 0])))


def test_linear_spline():
    """ Test linear spline forward. """
    inputs = np.array(np.random.random((100, 10)), dtype=np.float64)
    pdf = np.ones((100, 10, 10), dtype=np.float64)

    output, logabsdet = linear_spline(inputs, pdf)

    assert np.all(output >= 0)
    assert np.all(output <= 1)
    assert not np.any(np.isnan(logabsdet))


def test_linear_spline_inverse():
    """ Test linear spline inverse. """
    inputs = np.array(np.random.random((100, 10)), dtype=np.float64)
    pdf = np.ones((100, 10, 10), dtype=np.float64)

    output, logabsdet = linear_spline(inputs, pdf, True)

    assert np.all(output >= 0)
    assert np.all(output <= 1)
    assert not np.any(np.isnan(logabsdet))


def test_quadratic_spline_throws():
    """ Test quadratic spline bin errors. """
    inputs = np.array(np.random.random((100, 10)), dtype=np.float64)
    widths = np.ones((100, 10, 10), dtype=np.float64)
    heights = np.ones((100, 10, 11), dtype=np.float64)

    with pytest.raises(ValueError):
        quadratic_spline(inputs, widths, heights, min_bin_width=0.5)

    with pytest.raises(ValueError):
        quadratic_spline(inputs, widths, heights, min_bin_height=0.5)


def test_quadratic_spline():
    """ Test quadratic spline forward. """
    inputs = np.array(np.random.random((100, 10)), dtype=np.float64)
    widths = np.ones((100, 10, 10), dtype=np.float64)
    heights = np.ones((100, 10, 11), dtype=np.float64)

    output, logabsdet = quadratic_spline(inputs, widths, heights)

    assert np.all(output >= 0)
    assert np.all(output <= 1)
    assert not np.any(np.isnan(logabsdet))


def test_quadratic_spline_boundary():
    """ Test quadratic spline boundary terms. """
    inputs = np.array(np.random.random((100, 10)), dtype=np.float64)
    widths = np.ones((100, 10, 10), dtype=np.float64)
    heights = np.ones((100, 10, 9), dtype=np.float64)

    output, logabsdet = quadratic_spline(inputs, widths, heights)

    assert np.all(output >= 0)
    assert np.all(output <= 1)
    assert not np.any(np.isnan(logabsdet))


def test_quadratic_spline_inverse():
    """ Test quadratic spline inverse. """
    inputs = np.array(np.random.random((100, 10)), dtype=np.float64)
    widths = np.ones((100, 10, 10), dtype=np.float64)
    heights = np.ones((100, 10, 11), dtype=np.float64)

    output, logabsdet = quadratic_spline(inputs, widths, heights, True)

    assert np.all(output >= 0)
    assert np.all(output <= 1)
    assert not np.any(np.isnan(logabsdet))


def test_quadratic_spline_boundary_inverse():
    """ Test quadratic spline inverse boundary terms. """
    inputs = np.array(np.random.random((100, 10)), dtype=np.float64)
    widths = np.ones((100, 10, 10), dtype=np.float64)
    heights = np.ones((100, 10, 9), dtype=np.float64)

    output, logabsdet = quadratic_spline(inputs, widths, heights, True)

    assert np.all(output >= 0)
    assert np.all(output <= 1)
    assert not np.any(np.isnan(logabsdet))


def test_cubic_spline_throws():
    """ Test cubic spline bin errors. """
    inputs = np.array(np.random.random((100, 10)), dtype=np.float64)
    widths = np.ones((100, 10, 10), dtype=np.float64)
    heights = np.ones((100, 10, 11), dtype=np.float64)
    deriv_left = np.ones((100, 10, 10), dtype=np.float64)
    deriv_right = np.ones((100, 10, 10), dtype=np.float64)

    with pytest.raises(ValueError):
        cubic_spline(inputs, widths, heights, deriv_left,
                     deriv_right, min_bin_width=0.5)

    with pytest.raises(ValueError):
        cubic_spline(inputs, widths, heights, deriv_left,
                     deriv_right, min_bin_height=0.5)


def test_cubic_spline():
    """ Test cubic spline forward. """
    inputs = np.array(np.random.random((100, 10)), dtype=np.float64)
    widths = np.array(np.random.random((100, 10, 10)), dtype=np.float64)
    heights = np.array(np.random.random((100, 10, 10)), dtype=np.float64)
    deriv_left = np.ones((100, 10, 1), dtype=np.float64)
    deriv_right = np.ones((100, 10, 1), dtype=np.float64)

    output, logabsdet = cubic_spline(
        inputs, widths, heights, deriv_left, deriv_right)

    assert np.all(output >= 0)
    assert np.all(output <= 1)
    assert not np.any(np.isnan(logabsdet))


def test_cubic_spline_inverse():
    """ Test cubic spline inverse. """
    inputs = np.array(np.random.random((100, 10)), dtype=np.float64)
    widths = np.array(np.random.random((100, 10, 10)), dtype=np.float64)
    heights = np.array(np.random.random((100, 10, 10)), dtype=np.float64)
    deriv_left = np.array(np.random.random((100, 10, 1)), dtype=np.float64)
    deriv_right = np.array(np.random.random((100, 10, 1)), dtype=np.float64)

    output, logabsdet = cubic_spline(
        inputs, widths, heights, deriv_left, deriv_right, True)

    assert np.all(output >= 0)
    assert np.all(output <= 1)
    assert not np.any(np.isnan(logabsdet))


def test_rational_quadratic_spline_throws():
    """ Test rational quadratic spline bin errors. """
    inputs = np.array(np.random.random((100, 10)), dtype=np.float64)
    widths = np.ones((100, 10, 10), dtype=np.float64)
    heights = np.ones((100, 10, 10), dtype=np.float64)
    derivatives = np.ones((100, 10, 10), dtype=np.float64)

    with pytest.raises(ValueError):
        rational_quadratic_spline(
            inputs, widths, heights, derivatives, min_bin_width=0.5)

    with pytest.raises(ValueError):
        rational_quadratic_spline(
            inputs, widths, heights, derivatives, min_bin_height=0.5)


def test_rational_quadratic_spline():
    """ Test rational quadratic spline forward. """
    inputs = np.array(np.random.random((100, 10)), dtype=np.float64)
    widths = np.ones((100, 10, 10), dtype=np.float64)
    heights = np.ones((100, 10, 10), dtype=np.float64)
    derivatives = np.ones((100, 10, 10), dtype=np.float64)

    output, logabsdet = rational_quadratic_spline(
        inputs, widths, heights, derivatives)

    assert np.all(output >= 0)
    assert np.all(output <= 1)
    assert not np.any(np.isnan(logabsdet))


def test_rational_quadratic_spline_inverse():
    """ Test rational quadratic spline inverse. """
    inputs = np.array(np.random.random((100, 10)), dtype=np.float64)
    widths = np.ones((100, 10, 10), dtype=np.float64)/10.0
    heights = np.ones((100, 10, 10), dtype=np.float64)/10.0
    derivatives = np.ones((100, 10, 10), dtype=np.float64)

    output, logabsdet = rational_quadratic_spline(
        inputs, widths, heights, derivatives, True)

    assert np.all(output >= 0)
    assert np.all(output <= 1)
    assert not np.any(np.isnan(logabsdet))
