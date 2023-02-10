"""
Configuration of the unittests. Set your directories here.
Pytest fixtures defined in this module are accessible to all unittest modules (starting on ``test_``).
"""
import math
import os
from collections import defaultdict

import pytest
from dimcat.analyzer.base import (
    Analyzer,
)
from dimcat import TPCrange, PitchClassVectors, ChordSymbolUnigrams, ChordSymbolBigrams, LocalKeySequence, LocalKeyUnique
from dimcat.data import Dataset, GroupedDataset, SlicedDataset
from dimcat.filter.base import Filter
from dimcat.filter import IsAnnotatedFilter
from dimcat.grouper.base import Grouper
from dimcat.grouper import (
    CorpusGrouper,
    ModeGrouper,
    PieceGrouper,
    YearGrouper,
)
from dimcat.pipeline import Pipeline
from dimcat.slicer.base import Slicer
from dimcat.slicer import (
    LocalKeySlicer,
    MeasureSlicer,
    NoteSlicer,
    PhraseSlicer,
)
from git import Repo
from ms3 import pretty_dict

# Directory holding your clones of DCMLab/unittest_metacorpus & DCMLab/pleyel_quartets
CORPUS_DIR = "~"


@pytest.fixture(
    scope="session",
    params=[
        ("pleyel_quartets", "10a25eb"),
        ("unittest_metacorpus", "51e4cb5"),
    ],
    ids=[
        "corpus",
        "metacorpus",
    ],
)
def small_corpora_path(request) -> str:
    """Compose the paths for the test corpora."""
    print("Path was requested")
    repo_name, test_commit = request.param
    path = os.path.join(CORPUS_DIR, repo_name)
    path = os.path.expanduser(path)
    assert os.path.isdir(path)
    repo = Repo(path)
    commit = repo.commit("HEAD")
    sha = commit.hexsha[: len(test_commit)]
    assert sha == test_commit
    return path


@pytest.fixture(scope="session")
def all_corpora_path() -> str:
    path = os.path.join(CORPUS_DIR, "all_subcorpora")
    return path


@pytest.fixture(
    scope="session",
    params=[
        (Dataset, True, False),
        #        (Dataset, False, True),
        #        (Dataset, True, True),
    ],
    ids=[
        "TSV only",
        #        "scores only",
        #        "TSV + scores"
    ],
)
def dataset(small_corpora_path, request) -> Dataset:
    path = small_corpora_path
    obj, tsv, scores = request.param
    initialized_obj = obj(directory=path, parse_tsv=tsv, parse_scores=scores)
    print(
        f"\nInitialized {type(initialized_obj).__name__}(directory='{path}', "
        f"parse_tsv={tsv}, parse_scores={scores})"
    )
    return initialized_obj


@pytest.fixture(
    params=[True, False],
    ids=["once_per_group", ""],
)
def once_per_group(request) -> bool:
    return request.param


@pytest.fixture(
    params=[
        TPCrange,
        PitchClassVectors,
        ChordSymbolUnigrams,
        ChordSymbolBigrams,
        LocalKeyUnique,
        LocalKeySequence,
    ],
    ids=[
        "TPCrange",
        "PitchClassVectors",
        "ChordSymbolUnigrams",
        "ChordSymbolBigrams",
        "LocalKeyUnique",
        "LocalKeySequence",
    ],
)
def analyzer(request) -> Analyzer:
    return request.param()


@pytest.fixture(
    scope="session",
    params=[
        PhraseSlicer(),
        MeasureSlicer(),
        NoteSlicer(4),
        # NoteSlicer(),
        LocalKeySlicer(),
    ],
    ids=[
        "PhraseSlicer",
        "MeasureSlicer",
        "NoteSlicer_whole",
        # "NoteSlicer_onsets",
        "LocalKeySlicer",
    ],
)
def slicer(request) -> Slicer:
    return request.param


@pytest.fixture(
    scope="session",
)
def apply_slicer(slicer, dataset) -> SlicedDataset:
    sliced_data = slicer.process_data(dataset)
    print(
        f"\n{len(dataset.indices)} indices before slicing, after: {len(sliced_data.indices)}"
    )
    assert len(sliced_data.sliced) > 0
    assert len(sliced_data.slice_info) > 0
    assert len(sliced_data.index_levels["indices"]) > 2
    assert len(sliced_data.index_levels["slicer"]) > 0
    for facet, sliced in sliced_data.sliced.items():
        grouped_by_piece = defaultdict(list)
        for id, chunk in sliced.items():
            assert chunk.index.nlevels == 1
            assert len(id) == 3
            piece_id = id[:2]
            grouped_by_piece[piece_id].extend(chunk.duration_qb.to_list())
        for piece_id, durations in grouped_by_piece.items():
            # test if the facet slices add up to the same duration as the original facet
            facet_duration = sliced_data.get_item(piece_id, facet).duration_qb
            adds_up = math.isclose(sum(durations), facet_duration.sum())
            if not adds_up:
                print(
                    f"{piece_id}: Durations for facet {facet} sum up to {facet_duration.sum()}, "
                    f"but the slices add up to {sum(durations)}"
                )
    return sliced_data


@pytest.fixture(
    scope="session",
)
def sliced_data(apply_slicer) -> SlicedDataset:
    return apply_slicer


@pytest.fixture(
    scope="session",
    params=[
        CorpusGrouper(),
        PieceGrouper(),
        YearGrouper(),
    ],
    ids=["CorpusGrouper", "PieceGrouper", "YearGrouper"],
)
def grouper(request) -> Grouper:
    return request.param


@pytest.fixture(
    scope="session",
)
def apply_grouper(grouper, dataset) -> GroupedDataset:
    grouped_data = grouper.process_data(dataset)
    return grouped_data


@pytest.fixture(
    scope="session",
)
def grouped_data(apply_grouper) -> GroupedDataset:
    return apply_grouper


@pytest.fixture(
    scope="session",
    params=[
        Pipeline([LocalKeySlicer(), ModeGrouper()]),
        Pipeline([IsAnnotatedFilter(), PhraseSlicer()]),
    ],
    ids=[
        "PL_ModeGrouper",
        "PL_Filtered_PhraseSlicer",
    ],
)
def pipeline(request, dataset) -> Pipeline:
    grouped_data = request.param.process_data(dataset)
    return grouped_data


@pytest.fixture(
    scope="session",
)
def pipelined_data(pipeline) -> Dataset:
    return pipeline


@pytest.fixture(
    scope="session",
    params=[
        IsAnnotatedFilter(),
    ],
    ids=["IsAnnotatedFilter"],
)
def filter(request, dataset) -> Filter:
    filtered_data = request.param.process_data(dataset)
    print(f"\n{pretty_dict(filtered_data.indices)}")
    return filtered_data
