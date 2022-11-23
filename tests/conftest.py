"""
    Dummy conftest.py for dimcat.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    - https://docs.pytest.org/en/stable/fixture.html
    - https://docs.pytest.org/en/stable/writing_plugins.html
"""
import math
import os
from collections import defaultdict

import pytest
from dimcat.analyzer import (
    ChordSymbolBigrams,
    ChordSymbolUnigrams,
    PitchClassVectors,
    TPCrange,
)
from dimcat.data import Dataset
from dimcat.filter import IsAnnotatedFilter
from dimcat.grouper import CorpusGrouper, ModeGrouper, PieceGrouper, YearGrouper
from dimcat.pipeline import Pipeline
from dimcat.slicer import LocalKeySlicer, MeasureSlicer, NoteSlicer, PhraseSlicer
from git import Repo
from ms3 import pretty_dict

# Directory holding your clones of DCMLab/unittest_metacorpus & DCMLab/pleyel_quartets
CORPUS_DIR = "~"
TEST_COMMIT = (
    "73ce500"  # commit of DCMLab/unittest_metacorpus for which the tests should pass
)


@pytest.fixture(
    scope="session",
    params=[
        ("pleyel_quartets", "12cac6e"),
        ("unittest_metacorpus", "73ce500"),
    ],
    ids=[
        "corpus",
        "metacorpus",
    ],
)
def small_corpora_path(request):
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
def all_corpora_path():
    path = os.path.join(CORPUS_DIR, "all_subcorpora")
    return path


@pytest.fixture(
    scope="session",
    params=[
        (Dataset, True, False),
        #        (Corpus, False, True),
        #        (Corpus, True, True),
    ],
    ids=[
        "TSV only",
        #        "scores only",
        #        "TSV + scores"
    ],
)
def corpus(small_corpora_path, request):
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
def once_per_group(request):
    return request.param


@pytest.fixture(
    params=[
        TPCrange,
        PitchClassVectors,
        ChordSymbolUnigrams,
        ChordSymbolBigrams,
    ],
    ids=["TPCrange", "PitchClassVectors", "ChordSymbolUnigrams", "ChordSymbolBigrams"],
)
def analyzer(once_per_group, request):
    return request.param(once_per_group=once_per_group)


@pytest.fixture(
    scope="session",
    params=[
        PhraseSlicer(),
        MeasureSlicer(),
        NoteSlicer(1),
        NoteSlicer(),
        LocalKeySlicer(),
    ],
    ids=[
        "PhraseSlicer",
        "MeasureSlicer",
        "NoteSlicer_quarters",
        "NoteSlicer_onsets",
        "LocalKeySlicer",
    ],
)
def slicer(request):
    return request.param


@pytest.fixture(
    scope="session",
)
def apply_slicer(slicer, corpus):
    sliced_data = slicer.process_data(corpus)
    print(
        f"\n{len(corpus.indices[()])} indices before slicing, after: {len(sliced_data.indices[()])}"
    )
    # assert len(sliced_data.sliced) > 0
    assert len(sliced_data.slice_info) > 0
    assert len(sliced_data.index_levels["indices"]) > 2
    # assert len(sliced_data.index_levels['slices']) > 0
    for group, index_group in corpus.indices.items():
        assert len(sliced_data.indices[group]) > len(index_group)
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
def sliced_data(apply_slicer):
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
def grouper(request):
    return request.param


@pytest.fixture(
    scope="session",
)
def apply_grouper(grouper, corpus):
    grouped_data = grouper.process_data(corpus)
    print(f"\n{pretty_dict(grouped_data.indices)}")
    assert () not in grouped_data.indices
    lengths = [len(index_list) for index_list in grouped_data.indices.values()]
    assert 0 not in lengths, "Grouper has created empty groups."
    return grouped_data


@pytest.fixture(
    scope="session",
)
def grouped_data(apply_grouper):
    return apply_grouper


@pytest.fixture(
    scope="session",
    params=[
        Pipeline([LocalKeySlicer(), ModeGrouper()]),
        Pipeline([IsAnnotatedFilter()]),
    ],
    ids=[
        "ModeGrouper",
        "IsAnnotatedFilter",
    ],
)
def pipeline(request, corpus):
    grouped_data = request.param.process_data(corpus)
    print(f"\n{pretty_dict(grouped_data.indices)}")
    return grouped_data


@pytest.fixture(
    scope="session",
)
def pipelined_data(pipeline):
    return pipeline


@pytest.fixture(
    scope="session",
    params=[
        IsAnnotatedFilter(),
    ],
    ids=["IsAnnotatedFilter"],
)
def filter(request, corpus):
    filtered_data = request.param.process_data(corpus)
    print(f"\n{pretty_dict(filtered_data.indices)}")
    return filtered_data
