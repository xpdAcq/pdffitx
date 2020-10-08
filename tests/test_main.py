import subprocess
from pathlib import Path

from pkg_resources import resource_filename


def test_cli_main():
    main_file = Path(resource_filename('pdfstream', 'main.py'))
    cp = subprocess.run(
        ['python', str(main_file), '--', '--help'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    assert cp.returncode == 0
