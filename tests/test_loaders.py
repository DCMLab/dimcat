import os

import pytest
from dimcat.steps import MuseScoreLoader

from tests.conftest import CORPUS_PATH


def get_score_paths(extensions=(".xml", ".musicxml", ".mxl")):
    paths = []
    for path, subdirs, files in os.walk(CORPUS_PATH):
        for file in files:
            fname, fext = os.path.splitext(file)
            if fext in extensions:
                paths.append(os.path.join(path, file))
    return paths


@pytest.fixture(
    params=get_score_paths(extensions=(".xml", ".musicxml", ".mxl")),
    ids=os.path.basename,
)
def score_path(request):
    return request.param


@pytest.fixture()
def tmp_package_path(request, tmp_path_factory):
    """Returns the path to the directory where serialized resources are stored."""
    name = request.node.name
    return str(tmp_path_factory.mktemp(name))


def test_musescore_loader(corpus_path, tmp_package_path):
    path = os.path.join(corpus_path, "mixed_files")
    print(f"{path} => {tmp_package_path}")
    L = MuseScoreLoader(
        source=path,
        basepath=tmp_package_path,
        only_metadata_fnames=False,
        exclude_re="changed",
    )
    assert os.path.dirname(L.descriptor_path) == tmp_package_path
    print(os.listdir(tmp_package_path))
    assert len(os.listdir(tmp_package_path)) > 1
