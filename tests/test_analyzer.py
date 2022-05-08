import pytest  # noqa: F401

__author__ = "Digital and Cognitive Musicology Lab"
__copyright__ = "École Polytechnique Fédérale de Lausanne"
__license__ = "GPL-3.0-or-later"


def test_analyzer(analyzer, corpus):
    data = analyzer.process_data(corpus)
    assert len(data.processed) > 0
    print(data.get(as_pandas=True))
