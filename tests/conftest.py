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

# Directory holding your clones of DCMLab/unittest_metacorpus & DCMLab/pleyel_quartets
CORPUS_DIR = "~"


@pytest.fixture(
    scope="session",
    params=["pleyel_quartets", "unittest_metacorpus"],
    ids=["single", "multiple"],
)
def corpus_path(request):
    """Compose the paths for the test corpora."""
    print("Path was requested")
    path = os.path.join(CORPUS_DIR, request.param)
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
def corpus(request, corpus_path):
    obj, tsv, scores = request.param
    initialized_obj = obj(directory=corpus_path, parse_tsv=tsv, parse_scores=scores)
    print(
        f"\nInitialized {type(initialized_obj).__name__}(directory='{corpus_path}', "
        f"parse_tsv={tsv}, parse_scores={scores})"
    )
    return initialized_obj


@pytest.fixture(
    scope="session",
    params=[
        TPCrange(),
        PitchClassVectors(),
        ChordSymbolUnigrams(),
        ChordSymbolBigrams(),
    ],
    ids=["TPCrange", "PitchClassVectors", "ChordSymbolUnigrams", "ChordSymbolBigrams"],
)
def analyzer(request):
    return request.param
