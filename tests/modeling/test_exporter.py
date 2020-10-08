from tempfile import TemporaryDirectory

import pytest

import pdffitx.modeling.exporter as exporter
from pdffitx.modeling.exporter import save


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stru_fmt": "cif"},
        {"stru_fmt": "xyz"}
    ]
)
def test_save(optimized_recipe, kwargs):
    with TemporaryDirectory() as temp_dir:
        res_file, fgr_files, stru_files = save(optimized_recipe, base_name="test", folder=temp_dir, **kwargs)
        assert res_file.is_file()
        assert len(fgr_files) == 1 and fgr_files[0].is_file()
        assert len(stru_files) == 1 and stru_files[0].is_file()


def test_write_crystal_error(db):
    with pytest.raises(ValueError):
        exporter.write_crystal(db['Ni_stru'], 'error.mol', fmt='mol')
