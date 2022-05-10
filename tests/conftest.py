"""
    Dummy conftest.py for dimcat.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    - https://docs.pytest.org/en/stable/fixture.html
    - https://docs.pytest.org/en/stable/writing_plugins.html
"""
import os

import pytest
from dimcat.analyzer import (
    ChordSymbolBigrams,
    ChordSymbolUnigrams,
    PitchClassVectors,
    TPCrange,
)
from dimcat.data import Corpus
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
    obj, tsv, scores = request.param
    initialized_obj = obj(
        directory=all_corpora_path, parse_tsv=tsv, parse_scores=scores
    )
    print(
        f"\nInitialized {type(initialized_obj).__name__}(directory='{all_corpora_path}', "
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
def slicer(request, corpus):
    sliced_data = request.param.process_data(corpus)
    print(f"\nBefore: {len(corpus.indices[()])}, after: {len(sliced_data.indices[()])}")
    assert len(sliced_data.sliced) > 0
    assert len(sliced_data.slice_info) > 0
    for group, index_group in corpus.indices.items():
        assert len(sliced_data.indices[group]) > len(index_group)
    return sliced_data


@pytest.fixture(
    scope="session",
)
def sliced_data(slicer):
    return slicer


@pytest.fixture(
    scope="session",
    params=[
        CorpusGrouper(),
        PieceGrouper(),
        YearGrouper(),
    ],
    ids=["CorpusGrouper", "PieceGrouper", "YearGrouper"],
)
def grouper(request, corpus):
    grouped_data = request.param.process_data(corpus)
    print(f"\n{pretty_dict(grouped_data.indices)}")
    assert () not in grouped_data.indices
    lengths = [len(index_list) for index_list in grouped_data.indices.values()]
    assert 0 not in lengths, "Grouper has created empty groups."
    return grouped_data


@pytest.fixture(
    scope="session",
)
def grouped_data(grouper):
    return grouper


@pytest.fixture(
    scope="session",
    params=[
        Pipeline([LocalKeySlicer(), ModeGrouper()]),
    ],
    ids=["ModeGrouper"],
)
def pipeline(request, corpus):
    grouped_data = request.param.process_data(corpus)
    print(f"\n{pretty_dict(grouped_data.indices)}")
    assert () not in grouped_data.indices
    return grouped_data


@pytest.fixture(
    scope="session",
)
def pipelined_data(pipeline):
    return pipeline
