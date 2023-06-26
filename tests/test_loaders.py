import os

import music21 as m21
import pytest
from dimcat.steps import MuseScoreLoader

# from dimcat.steps.loaders import Music21Loader
from dimcat.steps.loaders.m21 import Music21Score


def get_score_paths(extensions=(".xml", ".musicxml", ".mxl"), n=1):
    m21_path = m21.__path__[0]
    music21_corpus_path = os.path.join(m21_path, "corpus")
    paths = []
    i = 0
    for path, subdirs, files in os.walk(music21_corpus_path):
        if i == n:
            break
        for file in files:
            if i == n:
                break
            fname, fext = os.path.splitext(file)
            if fext in extensions:
                paths.append(os.path.join(path, file))
                i += 1
    return paths


@pytest.fixture(
    params=get_score_paths(
        extensions=(
            ".xml",
            ".musicxml",
        )
    ),
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


def test_music21_loader(corpus_path, score_path, tmp_package_path):
    M = Music21Score(score_path)
    M.parse()
    # L = Music21Loader(
    #     source=corpus_path,
    #     basepath=tmp_package_path,
    # )
    # L.process_resource(score_path)
    # assert len(L.processed_ids) == 1
