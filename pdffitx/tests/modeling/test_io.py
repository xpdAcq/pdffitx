import pytest

from pdffitx.io import load_parser


@pytest.mark.parametrize(
    "meta",
    [
        {},
        {"qdamp": 0.04, "qbroad": 0.02},
        {"qmin": 0.}
    ]
)
def test_load_data(db, meta):
    parser = load_parser(db["Ni_gr_file"], meta)
    assert parser
