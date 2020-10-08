import pytest
from diffpy.srfit.pdf import characteristicfunctions as F
from diffpy.structure import loadStructure
from pyobjcryst import loadCrystal

from pdffitx.modeling.adding import initialize
from pdffitx.modeling.creating import create
from pdffitx.modeling.fitobjs import MyParser
from pdffitx.modeling.main import multi_phase


@pytest.fixture
def data(db):
    parser = MyParser()
    parser.parseFile(db["Ni_gr_file"], {"qdamp": 0.04, "qbroad": 0.02})
    return parser


@pytest.fixture
def structures(db):
    return {"G0": db["Ni_stru"]}


@pytest.fixture
def functions():
    return {"f0": F.sphericalCF}


@pytest.fixture
def blank_recipe(data, structures, functions):
    return create("test", data, (2., 7., 0.1), "f0 * G0 + A * sin(r)", functions, structures)


@pytest.fixture
def filled_recipe(blank_recipe):
    initialize(blank_recipe)
    blank_recipe.A.setValue(0.)
    return blank_recipe


@pytest.fixture(params=[loadCrystal, loadStructure])
def recipe_two_strus(request, db):
    parser = MyParser()
    parser.parseFile(db["Ni_gr_file"], {"qdamp": 0.04, "qbroad": 0.02})
    stru = request.param(db["Ni_stru_file"])
    recipe = multi_phase([(F.sphericalCF, stru)], parser, fit_range=(2., 8.0, .1), values={
        'psize_G0': 200}, sg_params={'G0': 225})
    return recipe
