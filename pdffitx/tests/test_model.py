import pathlib

import diffpy.srfit.pdf.characteristicfunctions as F
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

import pdffitx.files as files
import pdffitx.io as io
import pdffitx.model as mod

plt.ioff()


def test_MultiPhaseModel_1(tmpdir):
    # create a model
    Ni = io.load_crystal(files.NI_CIF_FILE)

    def f(r, psize):
        return F.sphericalCF(r, psize)

    model = mod.MultiPhaseModel("f * Ni", {"Ni": Ni}, {"f": f})

    # load data and check if metadata is correctly loaded
    metadata = {"qdamp": 0.04, "qbroad": 0.02, "qmax": 24.0, "qmin": 0.0}
    profile = io.load_profile(files.NI_GR_FILE, metadata)
    for k, v in metadata.items():
        assert profile.meta[k] == v
    model.set_profile(profile)
    gs = model.get_generators()
    for g in gs.values():
        assert g.qdamp.getValue() == metadata["qdamp"]
        assert g.qbroad.getValue() == metadata["qbroad"]
        assert g.getQmax() == metadata["qmax"]
        assert g.getQmin() == metadata["qmin"]

    # there shouldn't be any parameters named r and x
    with pytest.raises(KeyError):
        model.get_param("r")
    with pytest.raises(KeyError):
        model.get_param("x")

    # test the parameters added
    for p in ["f_psize", "Ni_scale", "Ni_a"]:
        assert model.get_param(p)

    # test set and get
    psize = model.get_param("f_psize")
    model.set_value(f_psize=150.)
    assert psize.getValue() == 150.
    model.set_bound(f_psize=(0., 200.))
    assert psize.bounds == [0., 200.]
    model.set_rel_bound(Ni_a=(0.2, 0.15))
    a = model.get_param("Ni_a")
    assert a.bounds == [a.getValue() - 0.2, a.getValue() + 0.15]

    # optimize the recipe
    model.set_xrange(2., 12., 0.1)
    model.set_options(ftol=1e-2)
    assert model.get_options() == {"ftol": 1e-2}
    order = [["Ni_scale", "f_psize"], "Ni_a", ["Ni_adp", "Ni_delta"]]
    model.set_order(*order)
    assert model.get_order() == order
    model.optimize()

    # check the expected values
    expected = {
        "f_psize": (198., 201.),
        "Ni_scale": (0.36, 0.38),
        "Ni_delta2": (1.2, 1.3),
        "Ni_a": (3.52, 3.53),
        "Ni_Ni0_Biso": (0.3, 0.6)
    }
    for name, bound in expected.items():
        assert bound[0] < model.get_param(name).getValue() < bound[1]

    # output the results
    model.update()
    model.save_all(str(tmpdir), "test")
    td = pathlib.Path(str(tmpdir))
    assert td.joinpath("test.txt").is_file()
    assert td.joinpath("test_result.nc").is_file()
    assert td.joinpath("test_fits.nc").is_file()
    assert td.joinpath("test_Ni.cif").is_file()

    # plot the fits
    model.plot()
    plt.show(block=False)
    plt.clf()

    # plot the exported fits
    ds = model.export_fits()
    fig, ax = plt.subplots()
    mod.plot_fits(ds, ax=ax)
    plt.show(block=False)
    plt.clf()

    # plot the stacked fits
    ds2 = xr.concat([ds, ds], dim="dim_0")
    mod.plot_fits_along_dim(ds2, "dim_0")
    plt.show(block=False)
    plt.clf()


def test_MultiPhaseModel_2(tmpdir):
    model = mod.MultiPhaseModel(equation="2 * r + a")
    assert model.get_equation() == "((2 * r) + a)"
    assert model.get_param("a")
    with pytest.raises(KeyError):
        model.get_param("r")
    with pytest.raises(KeyError):
        model.set_value(b=1)


def test_MultiPhaseModel_3():
    """Test the calc_phase method"""
    Ni = io.load_crystal(files.NI_CIF_FILE)
    model = mod.MultiPhaseModel("G", {"G": Ni})
    x = np.arange(0, 5, 0.1)
    arr = model.calc_phase(x, "G")
    assert isinstance(arr, xr.DataArray)


def test_MultiPhaseModel_4():
    """Test the eval method"""
    model = mod.MultiPhaseModel("a * x")
    model.set_value(a=1)
    x = np.linspace(0, 5, 6)
    y = np.zeros(6)
    model.set_data(x, y)
    model.eval()
    assert np.array_equal(model.get_profile().ycalc, x)


def test_MultiPhaseModel_5():
    """Test metadata related method"""
    model = mod.MultiPhaseModel("x")
    assert model.get_metadata() == {}
    model.set_metadata({"a": 1})
    assert model.get_metadata() == {"a": 1}


def test_MultiPhaseModel_6():
    """Test set equation method"""
    model = mod.MultiPhaseModel()
    model.set_equation("a * x")
    assert model.get_equation() == "(a * x)"


def test_fit_many_data():
    """Test the eval method"""
    model = mod.MultiPhaseModel("a * x")
    model.set_value(a=1)
    model.set_order("a")
    x = np.linspace(0., 1., 5)
    ds = xr.Dataset(
        {
            "ydata": (["cdata", "xdata"], np.stack([2. * x, 3. * x])),
            "zdata": (["cdata"], np.zeros((2,)))
        },
        {"xdata": x, "cdata": [0, 1]}
    )
    res, fits = model.fit_many_data(ds, "xdata", "ydata")
    print(res)
    print(fits)
