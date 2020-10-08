import pytest

from pdffitx.modeling import F
from pdffitx.modeling.fitfuncs import make_generator, get_sgpars, make_contribution, plot
from pdffitx.modeling.fitobjs import GenConfig, ConConfig, MyParser, FunConfig
from pdffitx.modeling.gens import GaussianGenerator


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ncpu": 1}
    ]
)
def test_make_generator(db, kwargs):
    gen_config = GenConfig("G0", db['Ni_stru'], **kwargs)
    make_generator(gen_config)


@pytest.mark.parametrize(
    "arg", ["P1"]
)
def test_get_sgpars(recipe_two_strus, arg):
    con = next(iter(recipe_two_strus.contributions.values()))
    gen = next(iter(con.generators.values()))
    get_sgpars(gen.phase, arg)


def test_get_sgpars_error(recipe_two_strus, db):
    with pytest.raises(ValueError):
        get_sgpars(db['Ni_stru'])


def test_make_contribution(db):
    parser = MyParser()
    parser.parseFile(db['Ni_gr_file'])
    gen_config = GenConfig("G", db['Ni_stru'])
    fun_config = FunConfig("f", F.sphericalCF)
    con_config = ConConfig(
        name="test",
        parser=parser,
        fit_range=(0, 8, 0.01),
        eq="G + B",
        funconfigs=[fun_config],
        genconfigs=[gen_config],
        baselines=[GaussianGenerator("B")]
    )
    con = make_contribution(con_config)
    assert len(con.generators) == 2


def test_plot(filled_recipe):
    filled_recipe.residual()
    plot(next(iter(filled_recipe.contributions.values())))
