import pytest  # noqa: F401

__author__ = "Digital and Cognitive Musicology Lab"
__copyright__ = "École Polytechnique Fédérale de Lausanne"
__license__ = "GPL-3.0-or-later"

from dimcat import ChordSymbolBigrams, LocalKeySlicer, TSVWriter


def assert_pipeline_dependency_raise(analyzer, data):
    if isinstance(analyzer, ChordSymbolBigrams):
        if not any(isinstance(step, LocalKeySlicer) for step in data.pipeline_steps):
            with pytest.raises(AssertionError):
                _ = analyzer.process_data(data)
            return True
    return False


def node_name2cfg_name(node_name):
    """Takes 'test_function[cfg_name]' and returns 'cfg_name'."""
    start_pos = node_name.index("[")
    caller_name = node_name[:start_pos]
    cfg_name = node_name[start_pos + 1 : -1]
    return caller_name, cfg_name


@pytest.fixture()
def analyzer_results(request):
    caller = request.node.originalname
    identifier = tuple(request.node.callspec._idlist)
    print(identifier)
    if caller == "test_analyzer":
        expected_results = {
            ("TSV only", "single", "TPCrange", "once_per_group"): {
                (): "pleyel_quartets-TPCrange"
            },
            ("TSV only", "single", "TPCrange", ""): {(): "pleyel_quartets-TPCrange"},
            ("TSV only", "single", "PitchClassVectors", "once_per_group"): {
                (): "pleyel_quartets-tpc-pcvs"
            },
            ("TSV only", "single", "PitchClassVectors", ""): {
                (): "pleyel_quartets-tpc-pcvs"
            },
            ("TSV only", "single", "ChordSymbolUnigrams", "once_per_group"): {
                (): "pleyel_quartets-ChordSymbolUnigrams"
            },
            ("TSV only", "single", "ChordSymbolUnigrams", ""): {
                (): "pleyel_quartets-ChordSymbolUnigrams"
            },
            ("TSV only", "multiple", "TPCrange", "once_per_group"): {
                (): "all-TPCrange"
            },
            ("TSV only", "multiple", "TPCrange", ""): {(): "all-TPCrange"},
            ("TSV only", "multiple", "PitchClassVectors", "once_per_group"): {
                (): "all-tpc-pcvs"
            },
            ("TSV only", "multiple", "PitchClassVectors", ""): {(): "all-tpc-pcvs"},
            ("TSV only", "multiple", "ChordSymbolUnigrams", "once_per_group"): {
                (): "all-ChordSymbolUnigrams"
            },
            ("TSV only", "multiple", "ChordSymbolUnigrams", ""): {
                (): "all-ChordSymbolUnigrams"
            },
        }
    if identifier in expected_results:
        return expected_results[identifier]
    return {}


def test_analyzer(analyzer, corpus, analyzer_results):
    assert len(corpus.index_levels["processed"]) == 0
    if assert_pipeline_dependency_raise(analyzer, corpus):
        return
    data = analyzer.process_data(corpus)
    print(f"{data.get()}")
    assert len(data.processed) > 0
    assert len(data.pipeline_steps) > 0
    automatic_filenames = TSVWriter(".").make_filenames(data)
    assert automatic_filenames == analyzer_results


def test_analyzing_slices(analyzer, sliced_data):
    if assert_pipeline_dependency_raise(analyzer, sliced_data):
        return
    data = analyzer.process_data(sliced_data)
    assert len(data.slice_info) > 0
    assert len(data.sliced) > 0
    for facet, sliced in data.sliced.items():
        for id, slices in sliced.items():
            assert slices.index.nlevels == 1
    analyzed_slices = data.get()
    print(f"{analyzed_slices}")
    # assert analyzed_slices.index.nlevels == 4


def test_analyzing_groups(analyzer, grouped_data):
    if assert_pipeline_dependency_raise(analyzer, grouped_data):
        return
    data = analyzer.process_data(grouped_data)
    assert () not in data.indices
    assert len(data.index_levels["groups"]) > 0
    print(f"{data.get()}")


def test_analyzing_pipelines(
    analyzer,
    pipelined_data,
):
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
