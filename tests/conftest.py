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
from dimcat.data import Corpus
from dimcat.filter import IsAnnotatedFilter
from dimcat.grouper import CorpusGrouper, ModeGrouper, PieceGrouper, YearGrouper
from dimcat.pipeline import Pipeline
from dimcat.slicer import LocalKeySlicer, NoteSlicer
from ms3 import pretty_dict

# Directory holding your clones of DCMLab/unittest_metacorpus & DCMLab/pleyel_quartets
CORPUS_DIR = "~"


@pytest.fixture(
    scope="session",
    params=[
        "pleyel_quartets",
        "unittest_metacorpus",
    ],
    ids=[
        "single",
        "multiple",
    ],
)
def small_corpora_path(request):
    """Compose the paths for the test corpora."""
    print("Path was requested")
    path = os.path.join(CORPUS_DIR, request.param)
    return path


@pytest.fixture(scope="session")
def all_corpora_path():
    path = os.path.join(CORPUS_DIR, "all_subcorpora")
    return path


@pytest.fixture(
    scope="session",
    params=[
        (Corpus, True, False),
        #        (Corpus, False, True),
        #        (Corpus, True, True),
    ],
    ids=[
        "TSV only",
        #        "scores only",
        #        "TSV + scores"
    ],
)
def corpus(all_corpora_path, request):
    path = all_corpora_path
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
        LocalKeySlicer(),
        NoteSlicer(),
    ],
    ids=[
        "LocalKeySlicer",
        "NoteSlicer",
    ],
)
def slicer(request):
    return request.param


@pytest.fixture(
    scope="session",
)
def apply_slicer(slicer, corpus):
    sliced_data = slicer.process_data(corpus)
    print(f"\nBefore: {len(corpus.indices[()])}, after: {len(sliced_data.indices[()])}")
    assert len(sliced_data.sliced) > 0
    assert len(sliced_data.slice_info) > 0
    assert len(sliced_data.index_levels["indices"]) > 2
    # assert len(sliced_data.index_levels['slices']) > 0
    for group, index_group in corpus.indices.items():
        assert len(sliced_data.indices[group]) > len(index_group)
    for facet, sliced in sliced_data.sliced.items():
        grouped_by_piece = defaultdict(list)
        for id, slices in sliced.items():
            assert slices.index.nlevels == 1
            assert len(id) == 3
            piece_id = id[:2]
            grouped_by_piece[piece_id].extend(slices.duration_qb.to_list())
        for piece_id, durations in grouped_by_piece.items():
            # test if the facet slices add up to the same duration as the original facet
            facet_duration = sliced_data.get_item(piece_id, facet).duration_qb.sum()
            assert math.isclose(sum(durations), facet_duration)
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
