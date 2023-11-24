"""
Configuring the test suite.
"""
import logging
import os
import platform
from typing import List, Optional

import frictionless as fl
import music21 as m21
import pytest
from _pytest.terminal import TerminalReporter
from dimcat.data.datasets.base import Dataset
from dimcat.data.packages.dc import DimcatPackage
from dimcat.data.resources.dc import DimcatResource
from dimcat.data.resources.utils import load_fl_resource
from dimcat.data.utils import make_rel_path
from dimcat.utils import scan_directory
from git import Repo

logger = logging.getLogger(__name__)


def pytest_terminal_summary(terminalreporter: TerminalReporter, exitstatus, config):
    terminalreporter.write_sep("=", "Versions summary", bold=True)
    import pandas as pd

    terminalreporter.write_line(f"Python version : {platform.python_version()}")
    terminalreporter.write_line(f"Pandas version : {pd.__version__}")


# ----------------------------- SETTINGS -----------------------------
# Directory holding your clone of github.com/DCMLab/unittest_metacorpus
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
logger.debug(f"TEST_DIR: {TEST_DIR!r}. Contents: {os.listdir(TEST_DIR)}")
CORPUS_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
logger.debug(f"CORPUS_DIR: {CORPUS_DIR!r}. Contents: {os.listdir(CORPUS_DIR)}")

# region test directories and files


def retrieve_and_check_corpus_path():
    """Compose the paths for the test corpora."""
    repo_name, test_commit = ("unittest_metacorpus", "aeebac1")
    path = os.path.join(CORPUS_DIR, repo_name)
    path = os.path.expanduser(path)
    assert os.path.isdir(path)
    repo = Repo(path)
    commit = repo.commit("HEAD")
    sha = commit.hexsha[: len(test_commit)]
    assert (
        sha == test_commit
    ), f"Your {path} is @ {sha}. Please do\n\tgit checkout {test_commit}."
    return path


CORPUS_PATH = retrieve_and_check_corpus_path()
RESOURCE_DESCRIPTOR_PATHS = {
    file: os.path.join(CORPUS_PATH, file)
    for file in os.listdir(CORPUS_PATH)
    if file.endswith(".resource.json")
}
PACKAGE_DESCRIPTOR_PATHS = {
    file: os.path.join(CORPUS_PATH, file)
    for file in os.listdir(CORPUS_PATH)
    if file.endswith("package.json")
}
TEST_N_SCORES = 3


@pytest.fixture(scope="session")
def corpus_path() -> str:
    return CORPUS_PATH


@pytest.fixture(scope="session")
def mixed_files_path(corpus_path) -> str:
    return os.path.join(corpus_path, "mixed_files")


def single_resource_descriptor_path() -> str:
    """Returns the path to a single resource."""
    return RESOURCE_DESCRIPTOR_PATHS["unittest_metacorpus.notes.resource.json"]


def datapackage_json_path() -> str:
    """Returns the path to a single resource."""
    return list(PACKAGE_DESCRIPTOR_PATHS.values())[0]


@pytest.fixture(scope="session")
def resource_descriptor_filename(resource_descriptor_path) -> str:
    """Returns the path to the descriptor file."""
    return make_rel_path(resource_descriptor_path, CORPUS_PATH)


@pytest.fixture(
    scope="session",
    params=RESOURCE_DESCRIPTOR_PATHS.values(),
    ids=list(RESOURCE_DESCRIPTOR_PATHS.keys()),
)
def resource_descriptor_path(request):
    return request.param


@pytest.fixture(
    scope="session",
    params=PACKAGE_DESCRIPTOR_PATHS.values(),
    ids=list(PACKAGE_DESCRIPTOR_PATHS.keys()),
)
def package_descriptor_path(request):
    return request.param


@pytest.fixture(scope="session")
def package_descriptor_filename(package_descriptor_path) -> str:
    """Returns the path to the descriptor file."""
    return os.path.basename(package_descriptor_path)


@pytest.fixture()
def tmp_serialization_path(request, tmp_path_factory):
    """Returns the path to the directory where serialized resources are stored."""
    if request.cls is None:
        name = request.function.__name__
    else:
        name = request.cls.__name__
    return str(tmp_path_factory.mktemp(name))


# endregion test directories and files
# region DimcatResource objects


@pytest.fixture(scope="session")
def fl_resource(resource_descriptor_path):
    """Returns a frictionless resource object."""
    return fl.Resource(resource_descriptor_path)


@pytest.fixture()
def assembled_resource(
    dataframe_from_fl_resource, fl_resource, tmp_serialization_path
) -> DimcatResource:
    resource = DimcatResource(basepath=tmp_serialization_path)
    resource.df = dataframe_from_fl_resource
    return resource


@pytest.fixture(scope="session")
def dataframe_from_fl_resource(fl_resource):
    """Returns a dataframe read directly from the normpath of the fl_resource."""
    return load_fl_resource(fl_resource)


@pytest.fixture()
def empty_resource():
    return DimcatResource()


@pytest.fixture()
def empty_resource_with_paths(tmp_serialization_path):
    return DimcatResource(basepath=tmp_serialization_path)


@pytest.fixture()
def resource_from_config(resource_from_descriptor):
    """Returns a DimcatResource object created from the descriptor on disk."""
    config = resource_from_descriptor.to_config()
    return DimcatResource.from_config(config)


@pytest.fixture()
def resource_from_dataframe(
    dataframe_from_fl_resource,
    fl_resource,
    tmp_serialization_path,
    resource_descriptor_filename,
) -> DimcatResource:
    """Returns a DimcatResource object created from the dataframe."""
    return DimcatResource.from_dataframe(
        df=dataframe_from_fl_resource,
        resource_name=fl_resource.name,
        basepath=tmp_serialization_path,
    )


@pytest.fixture()
def resource_from_descriptor(resource_descriptor_path):
    """Returns a DimcatResource object created from the descriptor on disk."""
    return DimcatResource.from_descriptor_path(descriptor_path=resource_descriptor_path)


@pytest.fixture()
def resource_from_dict(resource_from_descriptor):
    """Returns a DimcatResource object created from the descriptor source."""
    as_dict = resource_from_descriptor.to_dict()
    return DimcatResource.from_dict(as_dict)


@pytest.fixture()
def resource_from_fl_resource(
    fl_resource, resource_descriptor_filename
) -> DimcatResource:
    """Returns a Dimcat resource object created from the frictionless.Resource object."""
    return DimcatResource(
        resource=fl_resource, descriptor_filename=resource_descriptor_filename
    )


@pytest.fixture()
def resource_from_frozen_resource(resource_from_descriptor):
    """Returns a DimcatResource object created from a frozen resource."""
    return DimcatResource.from_resource(resource_from_descriptor)


@pytest.fixture()
def resource_from_memory_resource(resource_from_dataframe):
    """Returns a DimcatResource object created from a frozen resource."""
    assert resource_from_dataframe._resource.path
    return DimcatResource.from_resource(resource_from_dataframe)


@pytest.fixture()
def schema_resource(fl_resource):
    """Returns a (empty) DimcatResource with a pre-set frictionless.Schema object."""
    resource = DimcatResource()
    resource.column_schema = fl_resource.schema
    return resource


@pytest.fixture()
def serialized_resource(resource_from_dataframe) -> DimcatResource:
    resource_from_dataframe.store_dataframe()
    return resource_from_dataframe


@pytest.fixture()
def zipped_resource_copied_from_dc_package(
    package_from_fl_package, package_descriptor_filename
) -> DimcatResource:
    dc_resources = package_from_fl_package.get_resources_by_regex("notes")
    return DimcatResource.from_resource(dc_resources[0])


@pytest.fixture()
def zipped_resource_from_fl_package(
    fl_package,
    package_descriptor_filename,
) -> DimcatResource:
    """Returns a DimcatResource object created from the dataframe."""
    fl_resource = fl_package.get_resource("unittest_metacorpus.notes")
    return DimcatResource(
        resource=fl_resource, descriptor_filename=package_descriptor_filename
    )


@pytest.fixture(
    params=[
        "assembled_resource",
        "empty_resource",
        "empty_resource_with_paths",
        "resource_from_config",
        "resource_from_dataframe",
        "resource_from_descriptor",
        "resource_from_dict",
        "resource_from_fl_resource",
        "resource_from_frozen_resource",
        "resource_from_memory_resource",
        "schema_resource",
        "serialized_resource",
        "zipped_resource_copied_from_dc_package",
        "zipped_resource_from_fl_package",
    ]
)
def resource_object(
    request,
    assembled_resource,
    empty_resource,
    empty_resource_with_paths,
    resource_from_config,
    resource_from_dataframe,
    resource_from_descriptor,
    resource_from_dict,
    resource_from_fl_resource,
    resource_from_frozen_resource,
    resource_from_memory_resource,
    schema_resource,
    serialized_resource,
    zipped_resource_copied_from_dc_package,
    zipped_resource_from_fl_package,
):
    yield request.getfixturevalue(request.param)


# endregion DimcatResource objects


@pytest.fixture()
def fl_package(package_descriptor_path) -> fl.Package:
    """Returns a frictionless package object."""
    return fl.Package(package_descriptor_path)


@pytest.fixture()
def package_from_fl_package(fl_package, package_descriptor_path) -> DimcatPackage:
    """Returns a DimcatPackage object."""
    basepath, descriptor_filename = os.path.split(package_descriptor_path)
    return DimcatPackage.from_descriptor(
        descriptor=fl_package,
        descriptor_filename=descriptor_filename,
        basepath=basepath,
    )


@pytest.fixture()
def dataset_from_single_package(package_descriptor_path):
    dataset = Dataset()
    dataset.load_package(package_descriptor_path)
    return dataset


def get_music21_corpus_path():
    m21_path = m21.__path__[0]
    music21_corpus_path = os.path.join(m21_path, "corpus")
    return music21_corpus_path


@pytest.fixture()
def music21_corpus_path():
    return get_music21_corpus_path()


def get_score_paths(
    directory, extensions: Optional[str] = None, n: int | float = TEST_N_SCORES
) -> List[str]:
    paths = []
    for i, path in enumerate(
        scan_directory(
            directory,
            extensions=extensions,
        )
    ):
        if i >= n:
            break
        paths.append(path)
    return paths


def get_m21_score_paths(
    extensions=(".xml", ".musicxml", ".mxl"), n: int | float = TEST_N_SCORES
) -> List[str]:
    music21_corpus_path = get_music21_corpus_path()
    return get_score_paths(music21_corpus_path, extensions=extensions, n=n)


@pytest.fixture(scope="session")
def list_of_m21_score_paths() -> List[str]:
    return get_m21_score_paths()


def get_musescore_score_paths(
    extensions=(".mscz", ".mscx"), n: int | float = TEST_N_SCORES
) -> List[str]:
    return get_score_paths(CORPUS_PATH, extensions=extensions, n=n)


@pytest.fixture(scope="session")
def list_of_musescore_score_paths() -> List[str]:
    return get_musescore_score_paths()


def get_mixed_score_paths(n=TEST_N_SCORES) -> List[str]:
    return get_m21_score_paths(n=n / 2) + get_musescore_score_paths(n=n / 2)


@pytest.fixture(scope="session")
def list_of_mixed_score_paths() -> List[str]:
    return get_mixed_score_paths()


# region deprecated

# OLD TEST SETUP FROM v0.3.0
# import math
# import os
# from collections import defaultdict
# from dimcat.analyzer import (
#     ChordSymbolBigrams,
#     ChordSymbolUnigrams,
#     PitchClassVectors,
#     TPCrange,
# )
# from dimcat.dataset import Dataset
# from dimcat.filter import IsAnnotatedFilter
# from dimcat.grouper import CorpusGrouper, ModeGrouper, PieceGrouper, YearGrouper
# from dimcat.pipeline import Pipeline
# from dimcat.slicer import LocalKeySlicer, MeasureSlicer, NoteSlicer, PhraseSlicer
# from git import Repo
# from ms3 import pretty_dict
# @pytest.fixture(
#     scope="session",
#     params=[
#         ("pleyel_quartets", "10a25eb"),
#         ("unittest_metacorpus", "2d922c7"),
#     ],
#     ids=[
#         "corpus",
#         "metacorpus",
#     ],
# )
# def small_corpora_path(request):
#     """Compose the paths for the test corpora."""
#     print("Path was requested")
#     repo_name, test_commit = request.param
#     path = os.path.join(CORPUS_DIR, repo_name)
#     path = os.path.expanduser(path)
#     assert os.path.isdir(path)
#     repo = Repo(path)
#     commit = repo.commit("HEAD")
#     sha = commit.hexsha[: len(test_commit)]
#     assert sha == test_commit
#     return path
#
#
# @pytest.fixture(scope="session")
# def all_corpora_path():
#     path = os.path.join(CORPUS_DIR, "all_subcorpora")
#     return path
#
#
# @pytest.fixture(
#     scope="session",
#     params=[
#         (Dataset, True, False),
#         #        (Corpus, False, True),
#         #        (Corpus, True, True),
#     ],
#     ids=[
#         "TSV only",
#         #        "scores only",
#         #        "TSV + scores"
#     ],
# )
# def corpus(small_corpora_path, request):
#     path = small_corpora_path
#     obj, tsv, scores = request.param
#     initialized_obj = obj(directory=path, parse_tsv=tsv, parse_scores=scores)
#     print(
#         f"\nInitialized {type(initialized_obj).__name__}(directory='{path}', "
#         f"parse_tsv={tsv}, parse_scores={scores})"
#     )
#     return initialized_obj
#
#
# @pytest.fixture(
#     params=[True, False],
#     ids=["once_per_group", ""],
# )
# def once_per_group(request):
#     return request.param
#
#
# @pytest.fixture(
#     params=[
#         TPCrange,
#         PitchClassVectors,
#         ChordSymbolUnigrams,
#         ChordSymbolBigrams,
#     ],
#     ids=["TPCrange", "PitchClassVectors", "ChordSymbolUnigrams", "ChordSymbolBigrams"],
# )
# def analyzer(once_per_group, request):
#     return request.param(once_per_group=once_per_group)
#
#
# @pytest.fixture(
#     scope="session",
#     params=[
#         PhraseSlicer(),
#         MeasureSlicer(),
#         NoteSlicer(1),
#         NoteSlicer(),
#         LocalKeySlicer(),
#     ],
#     ids=[
#         "PhraseSlicer",
#         "MeasureSlicer",
#         "NoteSlicer_quarters",
#         "NoteSlicer_onsets",
#         "LocalKeySlicer",
#     ],
# )
# def slicer(request):
#     return request.param
#
#
# @pytest.fixture(
#     scope="session",
# )
# def apply_slicer(slicer, corpus):
#     sliced_data = slicer.process_data(corpus)
#     print(
#         f"\n{len(corpus.indices[()])} indices before slicing, after: {len(sliced_data.indices[()])}"
#     )
#     assert len(sliced_data.sliced) > 0
#     assert len(sliced_data.slice_info) > 0
#     assert len(sliced_data.index_levels["indices"]) > 2
#     assert len(sliced_data.index_levels["slicer"]) > 0
#     for facet, sliced in sliced_data.sliced.items():
#         grouped_by_piece = defaultdict(list)
#         for id, chunk in sliced.items():
#             assert chunk.index.nlevels == 1
#             assert len(id) == 3
#             piece_id = id[:2]
#             grouped_by_piece[piece_id].extend(chunk.duration_qb.to_list())
#         for piece_id, durations in grouped_by_piece.items():
#             # test if the facet slices add up to the same duration as the original facet
#             facet_duration = sliced_data.get_item(piece_id, facet).duration_qb
#             adds_up = math.isclose(sum(durations), facet_duration.sum())
#             if not adds_up:
#                 print(
#                     f"{piece_id}: Durations for facet {facet} sum up to {facet_duration.sum()}, "
#                     f"but the slices add up to {sum(durations)}"
#                 )
#     return sliced_data
#
#
# @pytest.fixture(
#     scope="session",
# )
# def sliced_data(apply_slicer):
#     return apply_slicer
#
#
# @pytest.fixture(
#     scope="session",
#     params=[
#         CorpusGrouper(),
#         PieceGrouper(),
#         YearGrouper(),
#     ],
#     ids=["CorpusGrouper", "PieceGrouper", "YearGrouper"],
# )
# def grouper(request):
#     return request.param
#
#
# @pytest.fixture(
#     scope="session",
# )
# def apply_grouper(grouper, corpus):
#     grouped_data = grouper.process_data(corpus)
#     print(f"\n{pretty_dict(grouped_data.indices)}")
#     assert () not in grouped_data.indices
#     lengths = [len(index_list) for index_list in grouped_data.indices.values()]
#     assert 0 not in lengths, "Grouper has created empty groups."
#     return grouped_data
#
#
# @pytest.fixture(
#     scope="session",
# )
# def grouped_data(apply_grouper):
#     return apply_grouper
#
#
# @pytest.fixture(
#     scope="session",
#     params=[
#         Pipeline([LocalKeySlicer(), ModeGrouper()]),
#         Pipeline([IsAnnotatedFilter()]),
#     ],
#     ids=[
#         "ModeGrouper",
#         "IsAnnotatedFilter",
#     ],
# )
# def pipeline(request, corpus):
#     grouped_data = request.param.process_data(corpus)
#     print(f"\n{pretty_dict(grouped_data.indices)}")
#     return grouped_data
#
#
# @pytest.fixture(
#     scope="session",
# )
# def pipelined_data(pipeline):
#     return pipeline
#
#
# @pytest.fixture(
#     scope="session",
#     params=[
#         IsAnnotatedFilter(),
#     ],
#     ids=["IsAnnotatedFilter"],
# )
# def filter(request, corpus):
#     filtered_data = request.param.process_data(corpus)
#     print(f"\n{pretty_dict(filtered_data.indices)}")
#     return filtered_data


# endregion deprecated
@pytest.fixture()
def tmp_package_path(request, tmp_path_factory):
    """Returns the path to the directory where serialized resources are stored."""
    name = request.node.name
    return str(tmp_path_factory.mktemp(name))
