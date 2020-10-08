import pytest

from pdffitx.modeling.adding import initialize


@pytest.mark.parametrize(
    "scale,delta,lat,adp,xyz,params,expect",
    [
        (
            True, None, None, None, None, None,
            {"G0_scale"}
        ),
        (
            False, "1", None, None, None, None,
            {"G0_delta1"}
        ),
        (
            False, "2", None, None, None, None,
            {"G0_delta2"}
        ),
        (
            False, None, "s", None, None, None,
            {"G0_a"}
        ),
        (
            False, None, "a", None, None, None,
            {"G0_a", "G0_b", "G0_c", "G0_alpha", "G0_beta", "G0_gamma"}
        ),
        (
            False, None, None, "e", None, None,
            {"G0_Ni_Biso"}
        ),
        (
            False, None, None, "a", None, None,
            {"G0_Ni0_Biso"}
        ),
        (
            False, None, None, "s", None, None,
            {"G0_Ni0_Biso"}
        ),
        (
            False, None, None, None, "s", None,
            set()
        ),
        (
            False, None, None, None, "a", None,
            {"G0_Ni0_x", "G0_Ni0_y", "G0_Ni0_z"}
        ),
        (
            False, None, None, None, None, "a",
            {"A", "f0_psize"}
        ),
        (
            False, None, None, None, None, ["A"],
            {"A"}
        ),
        pytest.param(
            False, "haha", None, None, None, None, set(),
            marks=pytest.mark.xfail
        ),
        pytest.param(
            False, None, "haha", None, None, None, set(),
            marks=pytest.mark.xfail
        ),
        pytest.param(
            False, None, None, "haha", None, None, set(),
            marks=pytest.mark.xfail
        ),
        pytest.param(
            False, None, None, None, "haha", None, set(),
            marks=pytest.mark.xfail
        ),
        pytest.param(
            False, None, None, None, None, "haha", set(),
            marks=pytest.mark.xfail
        ),
    ]
)
def test_initialize(blank_recipe, scale, delta, lat, adp, xyz, params, expect):
    initialize(blank_recipe, scale, delta, lat, adp, xyz, params)
    assert set(blank_recipe.getNames()) == expect
