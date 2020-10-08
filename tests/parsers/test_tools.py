import pytest

import pdfstream.parsers.tools as tools


@pytest.mark.parametrize(
    "dct,keys,expect",
    [
        ({"k": "v"}, (), {"k": "v"}),
        ({"k0": {"k00": "v0"}}, ("k0", "k00"), "v0")
    ]
)
def test_get_value(dct, keys, expect):
    assert tools.get_value(dct, keys) == expect
