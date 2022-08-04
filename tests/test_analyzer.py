import pytest  # noqa: F401

__author__ = "Digital and Cognitive Musicology Lab"
__copyright__ = "École Polytechnique Fédérale de Lausanne"
__license__ = "GPL-3.0-or-later"

from dimcat import ChordSymbolBigrams, LocalKeySlicer


def assert_pipeline_dependency_raise(analyzer, data):
    if isinstance(analyzer, ChordSymbolBigrams):
        if not any(isinstance(step, LocalKeySlicer) for step in data.pipeline_steps):
            with pytest.raises(AssertionError):
                _ = analyzer.process_data(data)
            return True
    return False


def test_analyzer(analyzer, corpus):
    assert len(corpus.index_levels["processed"]) == 0
    if assert_pipeline_dependency_raise(analyzer, corpus):
        return
    data = analyzer.process_data(corpus)
    assert len(data.processed) > 0
    assert len(data.pipeline_steps) > 0
    print(f"{data.get()}")


def test_analyzing_slices(analyzer, sliced_data):
    if assert_pipeline_dependency_raise(analyzer, sliced_data):
        return
    data = analyzer.process_data(sliced_data)
    assert len(data.slice_info) > 0
    assert len(data.sliced) > 0
    for facet, sliced in data.sliced.items():
        for id, slices in sliced.items():
            assert slices.index.nlevels == 1
    print(f"{data.get()}")


def test_analyzing_groups(analyzer, grouped_data):
    if assert_pipeline_dependency_raise(analyzer, grouped_data):
        return
    data = analyzer.process_data(grouped_data)
    assert () not in data.indices
    assert len(data.index_levels["groups"]) > 0
    print(f"{data.get()}")


def test_analyzing_pipelines(analyzer, pipelined_data):
    if assert_pipeline_dependency_raise(analyzer, pipelined_data):
        return
    data = analyzer.process_data(pipelined_data)
    print(f"{data.get()}")


def test_analyzing_grouped_pipelines(analyzer, pipelined_data, grouper):
    grouped = grouper.process_data(pipelined_data)
    for facet, slices in grouped.sliced.items():
        for id, df in slices.items():
            assert df.index.nlevels == 1
    if assert_pipeline_dependency_raise(analyzer, grouped):
        return
    data = analyzer.process_data(grouped)
    for facet, slices in data.sliced.items():
        for id, df in slices.items():
            assert df.index.nlevels == 1
    print(f"{data.get()}")
