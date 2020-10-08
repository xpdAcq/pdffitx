import numpy as np
import pytest

import pdfstream.modeling.fitobjs as fitobjs
from pdfstream.modeling.fitobjs import MyParser, GenConfig, ConConfig
from pdfstream.modeling.main import multi_phase


@pytest.mark.parametrize(
    "meta",
    [
        None,
        {'qmin': 1, 'qmax': 24, 'qdamp': 0.04, 'qbroad': 0.02}
    ]
)
def test_MyParser_parseDict(db, meta):
    parser = MyParser()
    parser.parseDict(db['Ni_gr'], meta=meta)
    recipe = multi_phase([db['Ni_stru']], parser, fit_range=(0., 8., .1))
    con = next(iter(recipe.contributions.values()))
    gen = next(iter(con.generators.values()))
    # if meta = None, generator will use the default values
    assert gen.getQmin() == parser._meta.get('qmin', 0.0)
    assert gen.getQmax() == parser._meta.get('qmax', 100. * np.pi)
    assert gen.qdamp.value == parser._meta.get('qdamp', 0.0)
    assert gen.qbroad.value == parser._meta.get('qbroad', 0.0)


@pytest.mark.parametrize(
    "data",
    [
        np.zeros((1, 5)),
        np.zeros((5, 5))
    ]
)
def test_MyParser_parseDict_error(data):
    parser = MyParser()
    with pytest.raises(ValueError):
        parser.parseDict(data)


@pytest.mark.parametrize(
    "data_key",
    ['Ni_pdfgetter']
)
@pytest.mark.parametrize(
    "meta",
    [None, {'qmax': 19}]
)
def test_MyParser_parsePDFGetter(db, data_key, meta):
    pdfgetter = db[data_key]
    parser = MyParser()
    parser.parsePDFGetter(pdfgetter, meta=meta)
    if meta:
        for key, value in meta.items():
            assert parser._meta[key] == value


@pytest.mark.parametrize(
    "data_key",
    ['Ni_gr_file']
)
@pytest.mark.parametrize(
    "meta",
    [None, {'qmax': 19}]
)
def test_MyParser_parseFile(db, data_key, meta):
    data_file = db[data_key]
    parser = MyParser()
    parser.parseFile(data_file, meta=meta)
    if meta:
        for key, value in meta.items():
            assert parser._meta[key] == value


@pytest.mark.parametrize(
    "mode,stype",
    [
        ("xray", "X"),
        ("neutron", "N"),
        ("sas", "X")
    ]
)
def test_map_stype(mode, stype):
    assert fitobjs.map_stype(mode) == stype


def test_map_stype_error():
    with pytest.raises(ValueError):
        fitobjs.map_stype("nray")


@pytest.mark.parametrize(
    "stru_key,expect",
    [
        ("Ni_stru_molecule", (False, True)),
        ("Ni_stru", (True, False)),
        ("Ni_stru_diffpy", (True, False))
    ]
)
def test_GenConfig(db, stru_key, expect):
    # noinspection PyArgumentList
    gen_config = GenConfig("G0", db[stru_key])
    assert gen_config.periodic == expect[0]
    assert gen_config.debye == expect[1]


@pytest.mark.parametrize(
    "kwargs,expect",
    [
        ({'res_eq': 'chiv'}, {'res_eq': 'chiv'})
    ]
)
def test_ConConfig(db, kwargs, expect):
    parser = MyParser()
    parser.parseFile(db['Ni_gr_file'])
    stru = db['Ni_stru']
    con_config = ConConfig(
        name="con",
        parser=parser,
        fit_range=(0., 8., .1),
        genconfigs=[GenConfig('G0', stru)],
        eq="G0",
        **kwargs
    )
    for key, value in expect.items():
        assert getattr(con_config, key) == value
