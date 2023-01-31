from dimcat.data import AnalyzedData
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
