import pytest
from tempfile import TemporaryDirectory

import pdffitx.modeling.exporter as exporter
from pdffitx.modeling.exporter import save
from pdffitx.modeling.main import optimize


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stru_fmt": "cif"},
        {"stru_fmt": "xyz"}
    ]
)
def test_save(recipe, kwargs):
    optimize(recipe, ['scale_G0', 'lat_G0'], xtol=1e-2, gtol=1e-2, ftol=1e-2)
    with TemporaryDirectory() as temp_dir:
        res_file, fgr_files, stru_files = save(recipe, base_name="test", folder=temp_dir, **kwargs)
        assert res_file.is_file()
        assert len(fgr_files) == 1 and fgr_files[0].is_file()
        assert len(stru_files) == 1 and stru_files[0].is_file()


def test_write_crystal_error(db):
    with pytest.raises(ValueError):
        exporter.write_crystal(db['Ni_stru'], 'error.mol', fmt='mol')
