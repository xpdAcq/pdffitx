"""Configuration of pytest."""
import numpy
import pyFAI
import pytest
from diffpy.pdfgetx import PDFConfig, PDFGetter
from diffpy.structure import loadStructure
from pdfstream.io import load_img, load_array
from pkg_resources import resource_filename
from pyobjcryst import loadCrystal
from pyobjcryst.molecule import Molecule

from pdffitx.io import load_parser
# data file
from pdffitx.modeling import create, initialize, optimize, F, multi_phase
from pdffitx.parsers import recipe_to_dict2

NI_PONI_FILE = resource_filename('tests', 'test_data/Ni_poni_file.poni')
NI_GR_FILE = resource_filename('tests', 'test_data/Ni_gr_file.gr')
NI_CHI_FILE = resource_filename('tests', 'test_data/Ni_chi_file.chi')
NI_FGR_FILE = resource_filename('tests', 'test_data/Ni_fgr_file.fgr')
NI_IMG_FILE = resource_filename('tests', 'test_data/Ni_img_file.tiff')
MASK_FILE = resource_filename("tests", "test_data/mask_file.npy")
KAPTON_IMG_FILE = resource_filename('tests', 'test_data/Kapton_img_file.tiff')
BLACK_IMG_FILE = resource_filename('tests', 'test_data/black_img.tiff')
WHITE_IMG_FILE = resource_filename('tests', 'test_data/white_img.tiff')
NI_IMG = load_img(NI_IMG_FILE)
KAPTON_IMG = load_img(KAPTON_IMG_FILE)
NI_GR = load_array(NI_GR_FILE)
NI_CHI = load_array(NI_CHI_FILE)
NI_FGR = load_array(NI_FGR_FILE)
NI_CONFIG = PDFConfig()
NI_CONFIG.readConfig(NI_GR_FILE)
NI_PDFGETTER = PDFGetter(NI_CONFIG)
AI = pyFAI.load(NI_PONI_FILE)
MASK = numpy.load(MASK_FILE)
BLACK_IMG = load_img(BLACK_IMG_FILE)
WHITE_IMG = load_img(WHITE_IMG_FILE)
# model file
ZRP_CIF_FILE = resource_filename('tests', 'test_data/ZrP_cif_file.cif')
NI_CIF_FILE = resource_filename("tests", "test_data/Ni_cif_file.cif")
NI_CRYSTAL = loadCrystal(NI_CIF_FILE)
ZRP_CRYSTAL = loadCrystal(ZRP_CIF_FILE)
NI_DIFFPY = loadStructure(NI_CIF_FILE)
NI_MOLECULE = Molecule(NI_CRYSTAL)

DB = {
    'Ni_img_file': NI_IMG_FILE,
    'Ni_img': NI_IMG,
    'Kapton_img_file': KAPTON_IMG_FILE,
    'Kapton_img': KAPTON_IMG,
    'Ni_poni_file': NI_PONI_FILE,
    'Ni_gr_file': NI_GR_FILE,
    'Ni_chi_file': NI_CHI_FILE,
    'Ni_fgr_file': NI_FGR_FILE,
    'ai': AI,
    'Ni_gr': NI_GR,
    'Ni_chi': NI_CHI,
    'Ni_fgr': NI_FGR,
    'black_img_file': BLACK_IMG_FILE,
    'white_img_file': WHITE_IMG_FILE,
    'black_img': BLACK_IMG,
    'white_img': WHITE_IMG,
    'Ni_config': NI_CONFIG,
    'Ni_pdfgetter': NI_PDFGETTER,
    'mask_file': MASK_FILE,
    'mask': MASK,
    'Ni_stru_file': NI_CIF_FILE,
    'Ni_stru': NI_CRYSTAL,
    'Ni_stru_molecule': NI_MOLECULE,
    'Ni_stru_diffpy': NI_DIFFPY,
    'ZrP_stru': ZRP_CRYSTAL
}


@pytest.fixture(scope="session")
def db():
    return DB


@pytest.fixture
def data():
    return load_parser(NI_GR_FILE, {"qdamp": 0.04, "qbroad": 0.02})


@pytest.fixture
def structures():
    return {"G0": NI_CRYSTAL}


@pytest.fixture
def functions():
    return {"f0": F.sphericalCF}


@pytest.fixture
def blank_recipe(data, structures, functions):
    return create("test", data, (2., 7., 0.1), "f0 * G0 + A * sin(r)", functions, structures)


@pytest.fixture
def filled_recipe(blank_recipe):
    recipe = blank_recipe
    initialize(recipe)
    recipe.A.setValue(0.)
    return recipe


@pytest.fixture(scope="session")
def optimized_recipe():
    ni_data = load_parser(NI_GR_FILE, {"qdamp": 0.04, "qbroad": 0.02})
    recipe = create("test", ni_data, (2., 7., 0.1), "G0", {}, {"G0": NI_CRYSTAL})
    initialize(recipe)
    optimize(recipe, ["G0_scale"], xtol=1e-3, gtol=1e-3, ftol=1e-3)
    return recipe


@pytest.fixture(params=[loadCrystal, loadStructure])
def multi_recipe(request, db):
    parser = load_parser(NI_GR_FILE, {"qdamp": 0.04, "qbroad": 0.02})
    stru = request.param(NI_CIF_FILE)
    recipe = multi_phase([(F.sphericalCF, stru)], parser, fit_range=(2., 8.0, .1), values={
        'psize_G0': 200}, sg_params={'G0': 225})
    return recipe


@pytest.fixture(scope="session")
def doc2(optimized_recipe):
    return recipe_to_dict2(optimized_recipe)
