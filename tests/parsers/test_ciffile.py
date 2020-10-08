from pdfstream.parsers.ciffile import to_crystal
from pyobjcryst.crystal import Crystal


def test_to_crystal(doc2):
    real = to_crystal(doc2)
    assert isinstance(real, Crystal)
