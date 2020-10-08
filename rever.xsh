from rever.activity import activity


@activity
def conda_release():
    $PYTHON release.py
    conda build $REVER_DIR/recipe
    conda build purge


@activity
def build_docs():
    make -C docs html


@activity
def install():
    python -m pip install -e .


$PROJECT = 'pdffitx'
$ACTIVITIES = [
    'version_bump',
    'install',
    'changelog',
    'build_docs',
    'tag',
    'push_tag',
    'ghrelease',
    'ghpages',
    'pypi',
    'conda_release'
]

$VERSION_BUMP_PATTERNS = [
    ('pdffitx/__init__.py', '__version__\s*=.*', "__version__ = '$VERSION'"),
    ('setup.py', 'version\s*=.*,', "version='$VERSION',")
]

$CHANGELOG_FILENAME = 'CHANGELOG.rst'
$CHANGELOG_TEMPLATE = 'TEMPLATE.rst'
$TAG_REMOTE = 'git@github.com:st3107/pdffitx.git'

$GITHUB_ORG = 'st3107'
$GITHUB_REPO = 'pdffitx'

$SPHINX_HOST_DIR = 'docs/build'
$GHPAGES_REPO = 'git@github.com:st3107/pdffitx.git'
$GHPAGES_BRANCH = 'gh-pages'
$GHPAGES_COPY = (
    ('$SPHINX_HOST_DIR/html', '$GHPAGES_REPO_DIR'),
)
