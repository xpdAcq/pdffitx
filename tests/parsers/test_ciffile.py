from pprint import pprint

import pytest
from mongomock import MongoClient
from pyobjcryst.crystal import Crystal

from pdfstream.parsers.ciffile import cif_to_dict, to_crystal


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mmjson": True},
        {"mmjson": False}
    ]
)
def test_cif_to_dict(db, kwargs):
    client = MongoClient()
    coll = client.db.coll
    dcts = cif_to_dict(
        db["Ni_stru_file"],
        **kwargs
    )
    for dct in dcts:
        pprint(dct)
        # test mongo friendly
        coll.insert_one(dct)


def test_to_crystal(doc2):
    real = to_crystal(doc2)
    assert isinstance(real, Crystal)
