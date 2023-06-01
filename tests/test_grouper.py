from random import sample

import pytest
from dimcat.dataset.processed import GroupedDataset
from dimcat.groupers.base import CustomPieceGrouper


@pytest.fixture()
def custom_piece_grouper(dataset_from_single_package):
    md = dataset_from_single_package.get_metadata()
    piece_ids = md.index.to_list()
    piece_groups = {i: sample(piece_ids, 2) for i in range(3)}
    return CustomPieceGrouper.from_dict(piece_groups)


def test_custom_piece_grouper(custom_piece_grouper, dataset_from_single_package):
    grouped_df = custom_piece_grouper.process(dataset_from_single_package)
    assert isinstance(grouped_df, GroupedDataset)
