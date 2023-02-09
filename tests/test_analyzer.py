import pandas as pd
import pytest  # noqa: F401
from dimcat import PhraseSlicer, TSVWriter
from ms3 import nan_eq

__author__ = "Digital and Cognitive Musicology Lab"
__copyright__ = "École Polytechnique Fédérale de Lausanne"
__license__ = "GPL-3.0-or-later"

from dimcat.utils import typestrings2types
from dimcat.data import GroupedData


def assert_pipeline_dependency_raise(analyzer_obj, data):
    """Checks if the given Analyzer can actually be applied to the given Dataset and returns False if yes (with
    the result that the test which called the function will continue).
    If it cannot, the function tests if a ValueError is thrown as expected and returns True (with the result that
    the test which called the function will stop). If no error is thrown, the test in question will fail."""
    analyzer_class = analyzer_obj.__class__
    analyzer_name = analyzer_class.__name__
    if len(analyzer_class.assert_steps) > 0:
        assert_steps = typestrings2types(analyzer_class.assert_steps)
        for step in assert_steps:
            if not any(
                isinstance(previous_step, step) for previous_step in data.pipeline_steps
            ):
                with pytest.raises(ValueError):
                    _ = analyzer_obj.process_data(data)
                print(
                    f"{analyzer_name} correctly raised ValueError "
                    f"because no {step.__name__} had been previously applied."
                )
                return True
    if len(analyzer_class.assert_previous_step) > 0:
        assert_previous_step = typestrings2types(analyzer_class.assert_previous_step)
        previous_step = data.pipeline_steps[0]
        if not isinstance(previous_step, assert_previous_step):
            with pytest.raises(ValueError):
                _ = analyzer_obj.process_data(data)
            print(
                f"{analyzer_name} correctly raised ValueError "
                f"because {step.__name__} is not an instance of {analyzer_class.assert_previous_step}."
            )
            return True
    if len(analyzer_class.excluded_steps) > 0:
        excluded_steps = typestrings2types(analyzer_class.excluded_steps)
        for step in excluded_steps:
            if any(
                isinstance(previous_step, step) for previous_step in data.pipeline_steps
            ):
                with pytest.raises(ValueError):
                    _ = analyzer_obj.process_data(data)
                print(
                    f"{analyzer_name} correctly raised ValueError because {step.__name__} had been applied."
                )
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
            ("TSV only", "corpus", "TPCrange"): {
                (
                    "pleyel_quartets",
                    "b307op2n1a",
                ): "pleyel_quartets|b307op2n1a-TPCrange",
                (
                    "pleyel_quartets",
                    "b307op2n1b",
                ): "pleyel_quartets|b307op2n1b-TPCrange",
                (
                    "pleyel_quartets",
                    "b307op2n1c",
                ): "pleyel_quartets|b307op2n1c-TPCrange",
                (
                    "pleyel_quartets",
                    "b309op2n3a",
                ): "pleyel_quartets|b309op2n3a-TPCrange",
                (
                    "pleyel_quartets",
                    "b309op2n3b",
                ): "pleyel_quartets|b309op2n3b-TPCrange",
                (
                    "pleyel_quartets",
                    "b309op2n3c",
                ): "pleyel_quartets|b309op2n3c-TPCrange",
            },
            ("TSV only", "corpus", "PitchClassVectors"): {
                (
                    "pleyel_quartets",
                    "b307op2n1a",
                ): "pleyel_quartets|b307op2n1a-tpc-pcvs",
                (
                    "pleyel_quartets",
                    "b307op2n1b",
                ): "pleyel_quartets|b307op2n1b-tpc-pcvs",
                (
                    "pleyel_quartets",
                    "b307op2n1c",
                ): "pleyel_quartets|b307op2n1c-tpc-pcvs",
                (
                    "pleyel_quartets",
                    "b309op2n3a",
                ): "pleyel_quartets|b309op2n3a-tpc-pcvs",
                (
                    "pleyel_quartets",
                    "b309op2n3b",
                ): "pleyel_quartets|b309op2n3b-tpc-pcvs",
                (
                    "pleyel_quartets",
                    "b309op2n3c",
                ): "pleyel_quartets|b309op2n3c-tpc-pcvs",
            },
            ("TSV only", "corpus", "ChordSymbolUnigrams"): {
                (
                    "pleyel_quartets",
                    "b307op2n1a",
                ): "pleyel_quartets|b307op2n1a-ChordSymbolUnigrams",
                (
                    "pleyel_quartets",
                    "b307op2n1b",
                ): "pleyel_quartets|b307op2n1b-ChordSymbolUnigrams",
                (
                    "pleyel_quartets",
                    "b307op2n1c",
                ): "pleyel_quartets|b307op2n1c-ChordSymbolUnigrams",
                (
                    "pleyel_quartets",
                    "b309op2n3a",
                ): "pleyel_quartets|b309op2n3a-ChordSymbolUnigrams",
                (
                    "pleyel_quartets",
                    "b309op2n3b",
                ): "pleyel_quartets|b309op2n3b-ChordSymbolUnigrams",
                (
                    "pleyel_quartets",
                    "b309op2n3c",
                ): "pleyel_quartets|b309op2n3c-ChordSymbolUnigrams",
            },
            ("TSV only", "corpus", "ChordSymbolBigrams"): {},
            ("TSV only", "corpus", "LocalKeyUnique"): {},
            ("TSV only", "corpus", "LocalKeySequence"): {},
            ("TSV only", "metacorpus", "TPCrange"): {
                (
                    "ravel_piano",
                    "Ravel_-_Jeux_dEau",
                ): "ravel_piano|Ravel_-_Jeux_dEau-TPCrange",
                (
                    "ravel_piano",
                    "Ravel_-_Miroirs_I._Noctuelles",
                ): "ravel_piano|Ravel_-_Miroirs_I._Noctuelles-TPCrange",
                (
                    "ravel_piano",
                    "Ravel_-_Miroirs_II._Oiseaux_tristes",
                ): "ravel_piano|Ravel_-_Miroirs_II._Oiseaux_tristes-TPCrange",
                (
                    "ravel_piano",
                    "Ravel_-_Miroirs_III._Une_Barque_sur_l'ocean",
                ): "ravel_piano|Ravel_-_Miroirs_III._Une_Barque_sur_l'ocean-TPCrange",
                (
                    "ravel_piano",
                    "Ravel_-_Miroirs_IV._Alborada_del_gracioso",
                ): "ravel_piano|Ravel_-_Miroirs_IV._Alborada_del_gracioso-TPCrange",
                (
                    "sweelinck_keyboard",
                    "SwWV258_fantasia_cromatica",
                ): "sweelinck_keyboard|SwWV258_fantasia_cromatica-TPCrange",
                (
                    "wagner_overtures",
                    "WWV090_Tristan_01_Vorspiel-Prelude_Ricordi1888Floridia",
                ): "wagner_overtures|WWV090_Tristan_01_Vorspiel-Prelude_Ricordi1888Floridia-TPCrange",
                (
                    "wagner_overtures",
                    "WWV096-Meistersinger_01_Vorspiel-Prelude_SchottKleinmichel",
                ): "wagner_overtures|WWV096-Meistersinger_01_Vorspiel-Prelude_SchottKleinmichel-TPCrange",
            },
            ("TSV only", "metacorpus", "PitchClassVectors"): {
                (
                    "ravel_piano",
                    "Ravel_-_Jeux_dEau",
                ): "ravel_piano|Ravel_-_Jeux_dEau-tpc-pcvs",
                (
                    "ravel_piano",
                    "Ravel_-_Miroirs_I._Noctuelles",
                ): "ravel_piano|Ravel_-_Miroirs_I._Noctuelles-tpc-pcvs",
                (
                    "ravel_piano",
                    "Ravel_-_Miroirs_II._Oiseaux_tristes",
                ): "ravel_piano|Ravel_-_Miroirs_II._Oiseaux_tristes-tpc-pcvs",
                (
                    "ravel_piano",
                    "Ravel_-_Miroirs_III._Une_Barque_sur_l'ocean",
                ): "ravel_piano|Ravel_-_Miroirs_III._Une_Barque_sur_l'ocean-tpc-pcvs",
                (
                    "ravel_piano",
                    "Ravel_-_Miroirs_IV._Alborada_del_gracioso",
                ): "ravel_piano|Ravel_-_Miroirs_IV._Alborada_del_gracioso-tpc-pcvs",
                (
                    "sweelinck_keyboard",
                    "SwWV258_fantasia_cromatica",
                ): "sweelinck_keyboard|SwWV258_fantasia_cromatica-tpc-pcvs",
                (
                    "wagner_overtures",
                    "WWV090_Tristan_01_Vorspiel-Prelude_Ricordi1888Floridia",
                ): "wagner_overtures|WWV090_Tristan_01_Vorspiel-Prelude_Ricordi1888Floridia-tpc-pcvs",
                (
                    "wagner_overtures",
                    "WWV096-Meistersinger_01_Vorspiel-Prelude_SchottKleinmichel",
                ): "wagner_overtures|WWV096-Meistersinger_01_Vorspiel-Prelude_SchottKleinmichel-tpc-pcvs",
            },
            ("TSV only", "metacorpus", "ChordSymbolUnigrams"): {
                (
                    "ravel_piano",
                    "Ravel_-_Jeux_dEau",
                ): "ravel_piano|Ravel_-_Jeux_dEau-ChordSymbolUnigrams",
                (
                    "ravel_piano",
                    "Ravel_-_Miroirs_I._Noctuelles",
                ): "ravel_piano|Ravel_-_Miroirs_I._Noctuelles-ChordSymbolUnigrams",
                (
                    "ravel_piano",
                    "Ravel_-_Miroirs_II._Oiseaux_tristes",
                ): "ravel_piano|Ravel_-_Miroirs_II._Oiseaux_tristes-ChordSymbolUnigrams",
                (
                    "ravel_piano",
                    "Ravel_-_Miroirs_III._Une_Barque_sur_l'ocean",
                ): "ravel_piano|Ravel_-_Miroirs_III._Une_Barque_sur_l'ocean-ChordSymbolUnigrams",
                (
                    "ravel_piano",
                    "Ravel_-_Miroirs_IV._Alborada_del_gracioso",
                ): "ravel_piano|Ravel_-_Miroirs_IV._Alborada_del_gracioso-ChordSymbolUnigrams",
                (
                    "sweelinck_keyboard",
                    "SwWV258_fantasia_cromatica",
                ): "sweelinck_keyboard|SwWV258_fantasia_cromatica-ChordSymbolUnigrams",
                (
                    "wagner_overtures",
                    "WWV090_Tristan_01_Vorspiel-Prelude_Ricordi1888Floridia",
                ): "wagner_overtures|WWV090_Tristan_01_Vorspiel-Prelude_Ricordi1888Floridia-ChordSymbolUnigrams",
                (
                    "wagner_overtures",
                    "WWV096-Meistersinger_01_Vorspiel-Prelude_SchottKleinmichel",
                ): "wagner_overtures|WWV096-Meistersinger_01_Vorspiel-Prelude_SchottKleinmichel-ChordSymbolUnigrams",
            },
            ("TSV only", "metacorpus", "ChordSymbolBigrams"): {},
            ("TSV only", "metacorpus", "LocalKeyUnique"): {},
            ("TSV only", "metacorpus", "LocalKeySequence"): {},
        }
    return expected_results[identifier]


def test_analyzer(analyzer, dataset, analyzer_results):
    assert len(dataset.index_levels["processed"]) == 0
    if assert_pipeline_dependency_raise(analyzer, dataset):
        return
    data = analyzer.process_data(dataset)
    print(f"{data.get_result_object()}")
    assert len(data.processed) > 0
    assert len(data.pipeline_steps) > 0
    if isinstance(data, GroupedData):
        automatic_filenames = TSVWriter(".").make_group_filenames(data)
    else:
        automatic_filenames = TSVWriter(".").make_index_filenames(data)
    assert automatic_filenames == analyzer_results


def diff_between_series(old, new):
    """Compares the values of two pandas.Series and computes a diff."""
    old_l, new_l = len(old), len(new)
    greater_length = max(old_l, new_l)
    if old_l != new_l:
        print(f"Old length: {old_l}, new length: {new_l}")
        old_is_shorter = new_l == greater_length
        shorter = old if old_is_shorter else new
        missing_rows = abs(old_l - new_l)
        patch = pd.Series(["missing row"] * missing_rows)
        shorter = pd.concat([shorter, patch], ignore_index=True)
        if old_is_shorter:
            old = shorter
        else:
            new = shorter
    old.index.rename("old_ix", inplace=True)
    new.index.rename("new_ix", inplace=True)
    diff = [
        (i, o, j, n)
        for ((i, o), (j, n)) in zip(old.iteritems(), new.iteritems())
        if not nan_eq(o, n)
    ]
    n_diffs = len(diff)
    if n_diffs > 0:
        comparison = pd.DataFrame(diff, columns=["old_ix", "old", "new_ix", "new"])
        print(
            f"{n_diffs}/{greater_length} ({n_diffs / greater_length * 100:.2f} %) rows are "
            f"different{' (showing first 20)' if n_diffs > 20 else ''}:\n{comparison}\n"
        )
        for a, b in zip(comparison.old.values, comparison.new.values):
            print(a)
            print(b)
        return comparison
    return pd.DataFrame()


def test_analyzing_slices(analyzer, sliced_data):
    if assert_pipeline_dependency_raise(analyzer, sliced_data):
        return
    data = analyzer.process_data(sliced_data)
    assert len(data.slice_info) > 0
    assert len(data.sliced) > 0
    for facet, sliced in data.sliced.items():
        for id, chunk in sliced.items():
            assert chunk.index.nlevels == 1
            try:
                interval_lengths = pd.Series(chunk.index.length, index=chunk.index)
            except AttributeError:
                print(chunk)
                raise
            if isinstance(sliced_data.get_previous_pipeline_step(), PhraseSlicer):
                # Currently, this test would fail for cases such as I}{ because in the resulting slices the label will
                # appear three times:
                # 1. as last row of the phrase ended by this label: this one will have an index interval of 0 but the
                #    'duration_qb' of the I chord (this is what doesn't pass the test)
                # 2. as the first two rows of the slice started by this label: once with chord = NA, index interval 0
                #    and duration_qb = 0; once with phraseend = NA, and the chord label with its normal duration
                continue
            duration_column = chunk.duration_qb.astype(float)
            diff = diff_between_series(
                interval_lengths.round(5), duration_column.round(5)
            )
            if len(diff) > 0:
                print(
                    f"COMPARING DURATION OF INDEX INTERVALS WITH COLUMN 'duration_qb' failed for ID {id}:"
                )
                a = interval_lengths
                b = chunk.index.right - chunk.index.left
                eq = (a == b).all()
                print("index.length == right-left:", eq)
                print(
                    "indices of the incongruent 'duration_qb' values:",
                    diff.old_ix.to_list(),
                )
                assert False

    analyzed_slices = data.get_result_object()
    print(f"{analyzed_slices}")
    # assert analyzed_slices.index.nlevels == 4


def test_analyzing_groups(analyzer, grouped_data):
    if assert_pipeline_dependency_raise(analyzer, grouped_data):
        return
    data = analyzer.process_data(grouped_data)
    assert () not in data.indices
    assert len(data.index_levels["groups"]) > 0
    print(f"{data.get_result_object()}")


def test_analyzing_pipelines(
    analyzer,
    pipelined_data,
):
    if assert_pipeline_dependency_raise(analyzer, pipelined_data):
        return
    data = analyzer.process_data(pipelined_data)
    print(f"{data.get_result_object()}")


def test_analyzing_grouped_pipelines(analyzer, pipelined_data, grouper):
    grouped = grouper.process_data(pipelined_data)
    if assert_pipeline_dependency_raise(analyzer, grouped):
        return
    data = analyzer.process_data(grouped)
    for facet, slices in data.sliced.items():
        for id, df in slices.items():
            assert df.index.nlevels == 1
    print(f"{data.get_result_object()}")
