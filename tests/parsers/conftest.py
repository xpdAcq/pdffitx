import pytest

import pdffitx.io as io
import pdffitx.modeling as md
from pdffitx.parsers.fitrecipe import recipe_to_dict2


@pytest.fixture(scope="module")
def recipe(db):
    """A recipe of crystal in pyobjcryst."""
    parser = io.load_parser(db["Ni_gr_file"], meta={"qdamp": 0.04, "qbroad": 0.02})
    structure = io.load_crystal(db["Ni_stru_file"])
    recipe = md.create("test", parser, (2.2, 7.2, 0.1), "G", {}, {"G": structure})
    md.initialize(recipe)
    md.optimize(recipe, ["G_scale"], verbose=0, ftol=1e-3)
    return recipe


@pytest.fixture(scope="module")
def doc2(recipe_two_strus):
    return recipe_to_dict2(recipe_two_strus)
