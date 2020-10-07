"""Used in release procedure."""
import shutil
import subprocess
import sys
import yaml
from pathlib import Path

# package info
NAME = 'pdffitx'
VERSION = "0.1.0"
DESCRIPTION = "A python package to model atomic pair distribution function (PDF) based on diffpy-cmi."
AUTHOR = "Songsheng Tao"
AUTHOR_EMAIL = 'st3107@columbia.edu'
URL = 'https://github.com/st3107/pdffitx'
LICENSE = "BSD (3-clause)"

# conda info
REVER_DIR = Path("rever")
REQUIREMENTS = Path("requirements")
CONDA_CHANNEL_SOURCES = ["defaults", "diffpy", "conda-forge"]
CONDA_CHANNEL_TARGETS = ["st3107"]
LICENSE_FILE = "LICENSE"
GIT_ACCOUNT = "st3107"


def conda_recipe() -> None:
    """Make conda recipe."""
    # create a new director
    recipe_dir = REVER_DIR / "recipe"
    if not recipe_dir.is_dir():
        recipe_dir.mkdir()
    # create conda_build_config.yaml
    conda_build_config_yaml = recipe_dir / "conda_build_config.yaml"
    with conda_build_config_yaml.open("w") as f:
        yaml.safe_dump(conda_build_config(), f, sort_keys=False)
    # create meta.yaml
    meta_yaml = recipe_dir / "meta.yaml"
    with meta_yaml.open("w") as f:
        yaml.safe_dump(conda_meta(), f, sort_keys=False)
    # copy license
    shutil.copy(LICENSE_FILE, recipe_dir / LICENSE_FILE.name)
    return


def conda_build_config() -> dict:
    """Make the dictionary of conda build configuration."""
    return {
        "channel_sources": CONDA_CHANNEL_SOURCES,
        "channel_targets": CONDA_CHANNEL_TARGETS
    }


def conda_meta() -> dict:
    """Make the dictionary of conda meta information."""
    name = NAME
    version = VERSION
    git_account = GIT_ACCOUNT
    build = read_dependencies(REQUIREMENTS / "build.txt")
    run = read_dependencies(REQUIREMENTS / "run.txt")
    tar_file_name = rf"{name}-{version}.tar.gz"
    sha256 = get_hash(REVER_DIR / tar_file_name)
    return {
        "package": {
            "name": name.lower(),
            "version": version
        },
        "source": {
            "url": rf"http://github.com/{git_account}/{name}/releases/download/{version}/{tar_file_name}",
            "sha256": sha256
        },
        "build": {
            "noarch": "python",
            "number": 0,
            "script": r"{{ PYTHON }} -m pip install . --no-deps -vv"
        },
        "requirements": {
            "build": build,
            "run": run
        },
        "test": {
            "imports": [name]
        },
        "about": {
            "home": URL,
            "license": LICENSE,
            "license_file": LICENSE_FILE,
            "summary": DESCRIPTION,
            "dev_url": URL,
        }
    }


def get_hash(file_path: Path, hash_type: str = "sha256") -> str:
    """Use openssl to get the hash of a file."""
    cp = subprocess.run(
        ["openssl", "dgst", rf"-{hash_type}", str(file_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    if cp.returncode != 0:
        sys.exit(cp.stderr)
    return cp.stdout.decode('utf-8').split("= ")[1].strip('\n')


def read_dependencies(txt_file: Path) -> list:
    """Read name of the required packages."""
    with txt_file.open("r") as f:
        dependencies = [
            line for line in f.read().splitlines()
            if not line.startswith('#')
        ]
    return dependencies


if __name__ == "__main__":
    conda_recipe()
