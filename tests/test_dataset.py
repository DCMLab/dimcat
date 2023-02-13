# import pytest # noqa: F401
from dimcat.data.loader import DcmlLoader
from dimcat.dtypes import PieceIndex


def test_duplicating_dcml_loader(dataset):
    index_before = dataset.get_piece_index()
    ms3_parse = dataset.loaders[0].loader
    new_loader = DcmlLoader()
    new_loader.set_loader(ms3_parse)
    dataset.attach_loader(new_loader)
    index_after = dataset.get_piece_index()
    assert len(index_after) == len(index_before) * 2
    concatenated = PieceIndex(index_before.values + index_after.values)
    index_before.extend(index_after)
    index_before == concatenated
