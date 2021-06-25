import pathlib

import diffpy.srfit.pdf.characteristicfunctions as F

import pdffitx.files as files
import pdffitx.io as io
import pdffitx.model as mod


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

    # test the set methods
    psize = model.get_param("psize")
    model.set_param(psize=150.)
    assert psize.getValue() == 150.
    model.set_bound(psize=(0., 200.))
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


def test_MultiPhaseModel_2(tmpdir):
    model = mod.MultiPhaseModel()
    model.reset_equation("2 * x")
    assert model.get_equation() == "(2 * x)"
