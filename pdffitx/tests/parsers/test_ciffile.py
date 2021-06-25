from pyobjcryst.crystal import Crystal

from pdffitx.parsers.ciffile import to_crystal


def test_to_crystal(doc2):
    real = to_crystal(doc2)
    assert isinstance(real, Crystal)
