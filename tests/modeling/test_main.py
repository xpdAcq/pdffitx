import pytest
from matplotlib import pyplot as plt

from pdffitx.modeling import F, multi_phase, optimize, fit_calib, MyParser


@pytest.mark.parametrize(
    "data_key,kwargs,free_params,use_cf,expected",
    [
        (
            "Ni_stru",
            {
                'values': {'scale_G0': 0.1, 'a_G0': 3.42, 'Biso_Ni_G0': 0.07, 'psize_f0': 300,
                           'delta2_G0': 2.5},
                'bounds': {'scale_G0': [-1, 1], 'a_G0': [0, 6], 'Biso_Ni_G0': [0, 1], 'psize_f0': [2, 400],
                           'delta2_G0': [0, 5]}
            },
            True,
            True,
            {'scale_G0', 'a_G0', 'Biso_Ni_G0', 'psize_f0', 'delta2_G0'}
        ),
        (
            "Ni_stru_diffpy",
            {
                'values': {'scale_G0': 0.1, 'a_G0': 3.42, 'Biso_Ni_G0': 0.07, 'psize_f0': 300,
                           'delta2_G0': 2.5},
                'bounds': {'scale_G0': [-1, 1], 'a_G0': [0, 6], 'Biso_Ni_G0': [0, 1], 'psize_f0': [2, 400],
                           'delta2_G0': [0, 5]},
            },
            False,
            True,
            {'Biso_Ni_G0', 'a_G0', 'alpha_G0', 'b_G0', 'beta_G0', 'c_G0', 'gamma_G0', 'delta2_G0',
             'psize_f0', 'scale_G0'}
        ),
        (
            "ZrP_stru",
            dict(),
            False,
            True,
            {'Biso_O_G0', 'Biso_P_G0', 'Biso_Zr_G0', 'a_G0', 'b_G0', 'beta_G0', 'c_G0',
             'delta2_G0', 'psize_f0', 'scale_G0'}
        ),
        (
            "ZrP_stru",
            {
                'bounds': {'x_0_G0': [-2, 2]}
            },
            True,
            True,
            {'Biso_O_G0', 'Biso_P_G0', 'Biso_Zr_G0', 'a_G0', 'b_G0', 'beta_G0', 'c_G0', 'delta2_G0',
             'psize_f0', 'scale_G0', 'x_0_G0', 'x_1_G0', 'x_2_G0', 'x_3_G0', 'x_4_G0', 'x_5_G0', 'x_6_G0',
             'x_7_G0', 'x_8_G0', 'x_9_G0', 'y_0_G0', 'y_1_G0', 'y_2_G0', 'y_3_G0', 'y_4_G0',
             'y_5_G0', 'y_6_G0', 'y_7_G0', 'y_8_G0', 'y_9_G0', 'z_0_G0', 'z_1_G0', 'z_2_G0',
             'z_3_G0', 'z_4_G0', 'z_5_G0', 'z_6_G0', 'z_7_G0', 'z_8_G0', 'z_9_G0'}
        ),
        (
            "Ni_stru",
            {
                'cf_params': ['psize_f0'],
                'sg_params': dict()
            },
            True,
            True,
            {'psize_f0'}
        ),
        (
            "Ni_stru_diffpy",
            {
                'cf_params': list(),
                'sg_params': {'G0': 225}
            },
            True,
            True,
            {'scale_G0', 'a_G0', 'Biso_Ni_G0', 'delta2_G0'}
        ),
        (
            "Ni_stru_diffpy",
            {
                'sg_params': {'G0': 225}
            },
            True,
            False,
            {'scale_G0', 'a_G0', 'Biso_Ni_G0', 'delta2_G0'}
        ),
        (
            "Ni_stru",
            {},
            True,
            False,
            {'scale_G0', 'a_G0', 'Biso_Ni_G0', 'delta2_G0'}
        ),
        (
            "Ni_stru",
            {"add_eq": "A * exp(- B * r ** 2) * sin(C * r)"},
            True,
            False,
            {'scale_G0', 'a_G0', 'Biso_Ni_G0', 'delta2_G0', "A", "B", "C"}
        )
    ]
)
def test_multi_phase(db, data_key, kwargs, free_params, use_cf, expected):
    parser = MyParser()
    parser.parseFile(db['Ni_gr_file'])
    phase = (F.sphericalCF, db[data_key]) if use_cf else db[data_key]
    recipe = multi_phase(
        [phase], parser,
        fit_range=(2., 8.0, .1),
        **kwargs
    )
    # xyz is added as fixed variables, free them for testing purpose
    if free_params:
        recipe.free("all")
    # check parameters
    if expected:
        assert set(recipe.getNames()) == expected
    # check default values
    values = kwargs.get('values')
    if values:
        actual_values = dict(zip(recipe.getNames(), recipe.getValues()))
        for name, expected_value in values.items():
            assert actual_values[name] == expected_value
    # check bounds
    bounds = kwargs.get('bounds')
    if bounds:
        actual_bounds = dict(zip(recipe.getNames(), recipe.getBounds()))
        for name, expected_bound in bounds.items():
            assert actual_bounds[name] == expected_bound


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(tags=['scale_G0'], xtol=1e-2, gtol=1e-2, ftol=1e-2),
        dict(tags=[('scale_G0', 'lat_G0'), 'adp_G0'], xtol=1e-2, gtol=1e-2, ftol=1e-2),
        dict(tags=['scale_G0'], verbose=1, xtol=1e-2, gtol=1e-2, ftol=1e-2),
        dict(tags=[('scale_G0', 'lat_G0'), 'adp_G0'], verbose=1, xtol=1e-2, gtol=1e-2, ftol=1e-2)
    ]
)
def test_optimize(recipe_two_strus, kwargs):
    optimize(recipe_two_strus, **kwargs)


def test_fit_calib(db):
    parser = MyParser()
    parser.parseFile(db['Ni_gr_file'])
    fit_calib(db['Ni_stru'], parser, fit_range=(2., 8., .1))
    plt.clf()
