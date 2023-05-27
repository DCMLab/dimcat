# import pandas as pd
# import pytest  # noqa: F401
# from dimcat import ChordSymbolBigrams, LocalKeySlicer, PhraseSlicer, TSVWriter
# from ms3 import nan_eq
#
# __author__ = "Digital and Cognitive Musicology Lab"
# __copyright__ = "École Polytechnique Fédérale de Lausanne"
# __license__ = "GPL-3.0-or-later"
#
#
# def assert_pipeline_dependency_raise(analyzer, data):
#     if isinstance(analyzer, ChordSymbolBigrams):
#         if not any(isinstance(step, LocalKeySlicer) for step in data.pipeline_steps):
#             with pytest.raises(AssertionError):
#                 _ = analyzer.process_data(data)
#             return True
#     return False
#
#
# def node_name2cfg_name(node_name):
#     """Takes 'test_function[cfg_name]' and returns 'cfg_name'."""
#     start_pos = node_name.index("[")
#     caller_name = node_name[:start_pos]
#     cfg_name = node_name[start_pos + 1 : -1]
#     return caller_name, cfg_name
#
#
# @pytest.fixture()
# def analyzer_results(request):
#     caller = request.node.originalname
#     identifier = tuple(request.node.callspec._idlist)
#     print(identifier)
#     if caller == "test_analyzer":
#         expected_results = {
#             ("TSV only", "corpus", "TPCrange", "once_per_group"): {
#                 (): "pleyel_quartets-TPCrange"
#             },
#             ("TSV only", "corpus", "TPCrange", ""): {(): "pleyel_quartets-TPCrange"},
#             ("TSV only", "corpus", "PitchClassVectors", "once_per_group"): {
#                 (): "pleyel_quartets-tpc-pcvs"
#             },
#             ("TSV only", "corpus", "PitchClassVectors", ""): {
#                 (): "pleyel_quartets-tpc-pcvs"
#             },
#             ("TSV only", "corpus", "ChordSymbolUnigrams", "once_per_group"): {
#                 (): "pleyel_quartets-ChordSymbolUnigrams"
#             },
#             ("TSV only", "corpus", "ChordSymbolUnigrams", ""): {
#                 (): "pleyel_quartets-ChordSymbolUnigrams"
#             },
#             ("TSV only", "corpus", "ChordSymbolBigrams", "once_per_group"): {},
#             ("TSV only", "corpus", "ChordSymbolBigrams", ""): {},
#             ("TSV only", "metacorpus", "TPCrange", "once_per_group"): {
#                 (): "all-TPCrange"
#             },
#             ("TSV only", "metacorpus", "TPCrange", ""): {(): "all-TPCrange"},
#             ("TSV only", "metacorpus", "PitchClassVectors", "once_per_group"): {
#                 (): "all-tpc-pcvs"
#             },
#             ("TSV only", "metacorpus", "PitchClassVectors", ""): {(): "all-tpc-pcvs"},
#             ("TSV only", "metacorpus", "ChordSymbolUnigrams", "once_per_group"): {
#                 (): "all-ChordSymbolUnigrams"
#             },
#             ("TSV only", "metacorpus", "ChordSymbolUnigrams", ""): {
#                 (): "all-ChordSymbolUnigrams"
#             },
#             ("TSV only", "metacorpus", "ChordSymbolBigrams", "once_per_group"): {},
#             ("TSV only", "metacorpus", "ChordSymbolBigrams", ""): {},
#         }
#     return expected_results[identifier]
#
#
# def test_analyzer(analyzer, corpus, analyzer_results):
#     assert len(corpus.index_levels["processed"]) == 0
#     if assert_pipeline_dependency_raise(analyzer, corpus):
#         return
#     data = analyzer.process_data(corpus)
#     print(f"{data.get()}")
#     assert len(data.processed) > 0
#     assert len(data.pipeline_steps) > 0
#     automatic_filenames = TSVWriter(".").make_filenames(data)
#     assert automatic_filenames == analyzer_results
#
#
# def diff_between_series(old, new):
#     """Compares the values of two pandas.Series and computes a diff."""
#     old_l, new_l = len(old), len(new)
#     greater_length = max(old_l, new_l)
#     if old_l != new_l:
#         print(f"Old length: {old_l}, new length: {new_l}")
#         old_is_shorter = new_l == greater_length
#         shorter = old if old_is_shorter else new
#         missing_rows = abs(old_l - new_l)
#         patch = pd.Series(["missing row"] * missing_rows)
#         shorter = pd.concat([shorter, patch], ignore_index=True)
#         if old_is_shorter:
#             old = shorter
#         else:
#             new = shorter
#     old.index.rename("old_ix", inplace=True)
#     new.index.rename("new_ix", inplace=True)
#     diff = [
#         (i, o, j, n)
#         for ((i, o), (j, n)) in zip(old.iteritems(), new.iteritems())
#         if not nan_eq(o, n)
#     ]
#     n_diffs = len(diff)
#     if n_diffs > 0:
#         comparison = pd.DataFrame(diff, columns=["old_ix", "old", "new_ix", "new"])
#         print(
#             f"{n_diffs}/{greater_length} ({n_diffs / greater_length * 100:.2f} %) rows are "
#             f"different{' (showing first 20)' if n_diffs > 20 else ''}:\n{comparison}\n"
#         )
#         for a, b in zip(comparison.old.values, comparison.new.values):
#             print(a)
#             print(b)
#         return comparison
#     return pd.DataFrame()
#
#
# def test_analyzing_slices(analyzer, sliced_data):
#     if assert_pipeline_dependency_raise(analyzer, sliced_data):
#         return
#     data = analyzer.process_data(sliced_data)
#     assert len(data.slice_info) > 0
#     assert len(data.sliced) > 0
#     for facet, sliced in data.sliced.items():
#         for id, chunk in sliced.items():
#             assert chunk.index.nlevels == 1
#             try:
#                 interval_lengths = pd.Series(chunk.index.length, index=chunk.index)
#             except AttributeError:
#                 print(chunk)
#                 raise
#             if isinstance(sliced_data.get_previous_pipeline_step(), PhraseSlicer):
#                 # Currently, this test would fail for cases such as I}{ because in the resulting slices the label will
#                 # appear three times:
#                 # 1. as last row of the phrase ended by this label: this one will have an index interval of 0 but the
#                 #    'duration_qb' of the I chord (this is what doesn't pass the test)
#                 # 2. as the first two rows of the slice started by this label: once with chord = NA, index interval 0
#                 #    and duration_qb = 0; once with phraseend = NA, and the chord label with its normal duration
#                 continue
#             duration_column = chunk.duration_qb.astype(float)
#             diff = diff_between_series(
#                 interval_lengths.round(5), duration_column.round(5)
#             )
#             if len(diff) > 0:
#                 print(
#                     f"COMPARING DURATION OF INDEX INTERVALS WITH COLUMN 'duration_qb' failed for ID {id}:"
#                 )
#                 a = interval_lengths
#                 b = chunk.index.right - chunk.index.left
#                 eq = (a == b).all()
#                 print("index.length == right-left:", eq)
#                 print(
#                     "indices of the incongruent 'duration_qb' values:",
#                     diff.old_ix.to_list(),
#                 )
#                 assert False
#
#     analyzed_slices = data.get()
#     print(f"{analyzed_slices}")
#     # assert analyzed_slices.index.nlevels == 4
#
#
# def test_analyzing_groups(analyzer, grouped_data):
#     if assert_pipeline_dependency_raise(analyzer, grouped_data):
#         return
#     data = analyzer.process_data(grouped_data)
#     assert () not in data.indices
#     assert len(data.index_levels["groups"]) > 0
#     print(f"{data.get()}")
#
#
# def test_analyzing_pipelines(
#     analyzer,
#     pipelined_data,
# ):
#     if assert_pipeline_dependency_raise(analyzer, pipelined_data):
#         return
#     data = analyzer.process_data(pipelined_data)
#     print(f"{data.get()}")
#
#
# def test_analyzing_grouped_pipelines(analyzer, pipelined_data, grouper):
#     grouped = grouper.process_data(pipelined_data)
#     for facet, slices in grouped.sliced.items():
#         for id, df in slices.items():
#             assert df.index.nlevels == 1
#     if assert_pipeline_dependency_raise(analyzer, grouped):
#         return
#     data = analyzer.process_data(grouped)
#     for facet, slices in data.sliced.items():
#         for id, df in slices.items():
#             assert df.index.nlevels == 1
#     print(f"{data.get()}")
