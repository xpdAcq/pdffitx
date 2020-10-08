import numpy as np
import pytest

from pdfstream.modeling.setting import set_range, get_range, set_values, get_values, bound_ranges, \
    bound_windows, get_bounds


@pytest.mark.parametrize(
    "rmin, rmax, rstep, expect",
    [
        (0., 1., 0.1, 0),
        ("obs", "obs", "obs", 1),
        (None, None, None, 2)
    ]
)
def test_set_range(blank_recipe, rmin, rmax, rstep, expect):
    x = blank_recipe.test.profile.x.copy()
    set_range(blank_recipe, rmin, rmax, rstep)
    fit_range = get_range(blank_recipe)
    if expect == 0:
        expect_array = np.arange(rmin, rmax + rstep, rstep)
        assert np.array_equal(fit_range, expect_array)
    elif expect == 1:
        assert np.array_equal(fit_range, blank_recipe.test.profile.xobs)
    elif expect == 2:
        assert np.array_equal(fit_range, x)


@pytest.mark.parametrize(
    "values, ignore, expect",
    [
        ({"G0_scale": 2.}, False, [2.]),
        ({"G0_scale": 2., "G5_scale": 1.}, True, [2., None]),
        pytest.param(
            {"G5_scale": 1.}, False, [None],
            marks=pytest.mark.xfail(raises=ValueError)
        )
    ]
)
def test_set_values(filled_recipe, values, ignore, expect):
    set_values(filled_recipe, values, ignore=ignore)
    real = get_values(filled_recipe, values.keys())
    assert real == expect


@pytest.mark.parametrize(
    "bounds, ignore, ratio, expect",
    [
        ({"G0_scale": (0.2, 1.)}, False, False, [[0.2, 1.]]),
        ({"G0_scale": {"lb": 0.2, "ub": 1.}}, False, False, [[0.2, 1.]]),
        ({"G0_scale": (0.2, 1.)}, False, True, [[0.1, 0.5]]),
        ({"G0_scale": {"lb": 0.2, "ub": 1.}}, False, True, [[0.1, 0.5]]),
        ({"G5_scale": (0.2, 1.)}, True, False, [None]),
        pytest.param(
            {"G5_scale": (0.2, 1.)}, False, False, [None], marks=pytest.mark.xfail(raises=ValueError)
        )
    ]
)
def test_bound_ranges(filled_recipe, bounds, ignore, ratio, expect):
    filled_recipe.G0_scale.setValue(0.5)
    bound_ranges(filled_recipe, bounds, ignore=ignore, ratio=ratio)
    real = get_bounds(filled_recipe, bounds.keys())
    assert real == expect


@pytest.mark.parametrize(
    "bounds, ignore, ratio, expect",
    [
        ({"G0_scale": 0.2}, False, False, [[0.3, 0.7]]),
        ({"G0_scale": 0.2}, False, True, [[0.4, 0.6]]),
        ({"G0_scale": (0.2, 0.4)}, False, False, [[0.3, 0.9]]),
        ({"G0_scale": {"lr": 0.2, "ur": 0.4}}, False, False, [[0.3, 0.9]]),
        ({"G0_scale": (0.2, 0.4)}, False, True, [[0.4, 0.7]]),
        ({"G0_scale": {"lr": 0.2, "ur": 0.4}}, False, True, [[0.4, 0.7]]),
        ({"G5_scale": (0.2, 0.4)}, True, False, [None]),
        pytest.param(
            {"G5_scale": (0.2, 0.4)}, False, False, [None], marks=pytest.mark.xfail(raises=ValueError)
        )
    ]
)
def test_bound_windows(filled_recipe, bounds, ignore, ratio, expect):
    filled_recipe.G0_scale.setValue(0.5)
    bound_windows(filled_recipe, bounds, ignore=ignore, ratio=ratio)
    real = get_bounds(filled_recipe, bounds.keys())
    assert real == expect
