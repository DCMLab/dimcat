import pytest  # noqa: F401

__author__ = "Digital and Cognitive Musicology Lab"
__copyright__ = "École Polytechnique Fédérale de Lausanne"
__license__ = "GPL-3.0-or-later"


def test_analyzer(analyzer, corpus):
    data = analyzer.process_data(corpus)
    assert len(data.processed) > 0
    print(data.get(as_pandas=True))


def test_analyzing_slices(sliced_data, analyzer):
    data = analyzer.process_data(sliced_data)
    assert len(data.slice_info) > 0
    assert len(data.sliced) > 0
    print(data.get(as_pandas=True))
