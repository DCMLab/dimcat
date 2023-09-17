from random import sample

import pytest
from dimcat.data.datasets.processed import GroupedDataset
from dimcat.steps.groupers.base import CustomPieceGrouper


@pytest.fixture()
def custom_piece_grouper(dataset_from_single_package):
    md = dataset_from_single_package.get_metadata()
    piece_ids = md.index.to_list()
    piece_groups = {i: sample(piece_ids, 2) for i in range(3)}
    grouper = CustomPieceGrouper.from_grouping(piece_groups)
    grouper.features = "notes"
    return grouper


def test_custom_piece_grouper(custom_piece_grouper, dataset_from_single_package):
    grouped_df = custom_piece_grouper.process_data(dataset_from_single_package)
    assert isinstance(grouped_df, GroupedDataset)
