from dimcat.data import AnalyzedData, Dataset
from IPython.display import display
from ms3 import pretty_dict


def property_test_on_grouped_data(grpd_dt):
    assert hasattr(grpd_dt, "grouped_indices")
    idcs = grpd_dt.grouped_indices
    n_groups = len(idcs)
    n_idcs_per_group = {group: len(ids) for group, ids in idcs.items()}
    print(
        f"Grouped the {len(grpd_dt.indices)} IDs into {n_groups} groups:\n{pretty_dict(n_idcs_per_group)}"
    )
    assert () not in grpd_dt.grouped_indices
    assert 0 not in list(n_idcs_per_group.values()), "Grouper has created empty groups."
    for group, notes in grpd_dt.iter_grouped_facet("notes"):
        assert notes.index.nlevels > 2
        display(notes)


def test_properties(grouped_data):
    property_test_on_grouped_data(grouped_data)
    analyzed_grouped = AnalyzedData(grouped_data)
    property_test_on_grouped_data(analyzed_grouped)


def test_transitivity_with_slicers(grouped_data, slicer):
    sliced_grouped = slicer.process_data(grouped_data)
    test_grouper = grouped_data.get_previous_pipeline_step()
    reset_data = Dataset(grouped_data)
    sliced_data = slicer.process_data(reset_data)
    grouped_sliced = test_grouper.process_data(sliced_data)
    for (sl_gr_group, sl_gr_notes), (gr_sl_group, gr_sl_notes) in zip(
        sliced_grouped.iter_grouped_facet("notes"),
        grouped_sliced.iter_grouped_facet("notes"),
    ):
        assert sl_gr_group == gr_sl_group
        assert len(sl_gr_notes) == len(gr_sl_notes)
