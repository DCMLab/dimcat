import os

import music21 as m21
import pytest
from dimcat import Dataset, Pipeline
from dimcat.steps import MuseScoreLoader
from dimcat.steps.loaders import Music21Loader
from dimcat.steps.loaders.base import PackageLoader
from dimcat.steps.loaders.utils import scan_directory


def get_music21_corpus_path():
    m21_path = m21.__path__[0]
    music21_corpus_path = os.path.join(m21_path, "corpus")
    return music21_corpus_path


def get_score_paths(extensions=(".xml", ".musicxml", ".mxl"), n=10):
    music21_corpus_path = get_music21_corpus_path()
    paths = []
    for i, path in enumerate(scan_directory(music21_corpus_path)):
        if i == n:
            break
        paths.append(path)
    return paths


@pytest.fixture(
    params=get_score_paths(
        extensions=(
            ".xml",
            ".musicxml",
        ),
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


def test_musescore_loader(mixed_files_path, tmp_package_path):
    print(f"{mixed_files_path} => {tmp_package_path}")
    L = MuseScoreLoader(
        "mixed_files",
        basepath=tmp_package_path,
        source=mixed_files_path,
        only_metadata_fnames=False,
        exclude_re="changed",
    )
    L.create_datapackage()
    assert os.path.dirname(L.get_descriptor_path()) == tmp_package_path
    print(os.listdir(tmp_package_path))
    assert len(os.listdir(tmp_package_path)) > 1


def test_music21_single_resource(corpus_path, score_path, tmp_package_path):
    L = Music21Loader(
        "music21_single_resource",
        basepath=tmp_package_path,
        source=corpus_path,
    )
    L.process_resource(score_path)
    print(tmp_package_path)
    assert len(L.processed_ids) == 1


@pytest.fixture()
def list_of_score_paths():
    return get_score_paths()


def test_music21_list_of_paths(list_of_score_paths, tmp_package_path):
    L = Music21Loader(
        package_name="music21_corpus",
        basepath=tmp_package_path,
        source=list_of_score_paths,
    )
    L.create_datapackage()
    assert len(L.processed_ids) == len(list_of_score_paths)
    print(tmp_package_path)
    print(os.listdir(tmp_package_path))
    assert len(os.listdir(tmp_package_path)) > 1


def test_loading_into_dataset(mixed_files_path, list_of_score_paths, tmp_package_path):
    MS = MuseScoreLoader(
        "musescore",
        source=mixed_files_path,
        only_metadata_fnames=False,
        exclude_re="changed",
    )
    M21 = Music21Loader(
        package_name="music21",
        source=list_of_score_paths,
    )
    D = Dataset(basepath=tmp_package_path)
    PL = Pipeline([MS, M21])
    dataset_loaded = PL.process(D)
    print(os.listdir(tmp_package_path))
    assert len(os.listdir(tmp_package_path)) > 3
    assert dataset_loaded.inputs.package_names == ["musescore", "music21"]


def test_package_loader(corpus_path):
    L = PackageLoader(source=corpus_path)
    D = L.process(Dataset())
    print(D)
    assert len(D.inputs) == 1
    assert D.inputs.package_names == ["unittest_metacorpus"]
