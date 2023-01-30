from IPython.display import display
from ms3 import pretty_dict


def test_properties(grouped_data):
    assert hasattr(grouped_data, "grouped_indices")
    idcs = grouped_data.grouped_indices
    n_groups = len(idcs)
    n_idcs_per_group = {group: len(ids) for group, ids in idcs.items()}
    print(
        f"Grouped the {len(grouped_data.indices)} IDs into {n_groups} groups:\n{pretty_dict(n_idcs_per_group)}"
    )
    assert () not in grouped_data.grouped_indices
    assert 0 not in list(n_idcs_per_group.values()), "Grouper has created empty groups."
    for group, notes in grouped_data.iter_facet("notes"):
        assert notes.index.nlevels > 2
        display(notes)
