import pytest  # noqa: F401

__author__ = "Digital and Cognitive Musicology Lab"
__copyright__ = "École Polytechnique Fédérale de Lausanne"
__license__ = "GPL-3.0-or-later"


def test_analyzer(analyzer, corpus):
    assert len(corpus.index_levels["processed"]) == 0
    data = analyzer.process_data(corpus)
    assert len(data.processed) > 0
    assert len(data.applied_pipeline) > 0
    print(f"{data.get(as_pandas=True)}")


def test_analyzing_slices(analyzer, sliced_data):
    data = analyzer.process_data(sliced_data)
    assert len(data.slice_info) > 0
    assert len(data.sliced) > 0
    print(f"{data.get(as_pandas=True)}")


def test_analyzing_groups(analyzer, grouped_data):
    data = analyzer.process_data(grouped_data)
    assert () not in data.indices
    assert len(data.index_levels["groups"]) > 0
    print(f"{data.get(as_pandas=True)}")


def test_analyzing_pipelines(analyzer, pipelined_data):
    data = analyzer.process_data(pipelined_data)
    assert () not in data.indices
    assert len(data.index_levels["groups"]) > 0
    print(f"{data.get(as_pandas=True)}")
