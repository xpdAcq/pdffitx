import numpy as np
import pyFAI
import pytest

from pdfstream.io import load_dict_from_poni, load_ai_from_calib_result


@pytest.fixture(scope="module")
def expect_qi(db):
    ai = pyFAI.load(db["Ni_poni_file"])
    q, i = ai.integrate1d(db['black_img'], 1024, safe=False)
    return q, i


def test_load_dict_from_poni(db, expect_qi):
    config = load_dict_from_poni(db["Ni_poni_file"])
    ai = pyFAI.AzimuthalIntegrator()
    ai.set_config(config)
    q, i = ai.integrate1d(db['black_img'], 1024, safe=False)
    assert np.array_equal(q, expect_qi[0])
    assert np.array_equal(i, expect_qi[1])


def test_load_ai_from_calib_result(db, expect_qi):
    dct = load_dict_from_poni(db['Ni_poni_file'])
    ai = load_ai_from_calib_result(dct)
    q, i = ai.integrate1d(db['black_img'], 1024, safe=False)
    assert np.array_equal(q, expect_qi[0])
    assert np.array_equal(i, expect_qi[1])
