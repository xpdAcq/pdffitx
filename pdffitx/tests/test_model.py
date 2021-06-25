import pathlib

import diffpy.srfit.pdf.characteristicfunctions as F
import matplotlib.pyplot as plt
import pytest
import xarray as xr

import pdffitx.files as files
import pdffitx.io as io
import pdffitx.model as mod

plt.ioff()


def test_MultiPhaseModel_1(tmpdir):
    # create a model
    Ni = io.load_crystal(files.NI_CIF_FILE)

    def f(x, psize):
        return F.sphericalCF(x, psize)

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
    model.set_order("Ni_scale", "Ni_a", ["Ni_adp", "Ni_delta"])
    assert model.get_order() == ["Ni_scale", "Ni_a", ["Ni_adp", "Ni_delta"]]
    model.optimize()

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
