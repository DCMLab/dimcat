from functools import lru_cache
from typing import Any

import dimcat.data as data_module
import pytest
from dimcat.data import (
    AnalyzedData,
    AnalyzedDataset,
    AnalyzedGroupedDataset,
    AnalyzedGroupedSlicedDataset,
    AnalyzedSlicedDataset,
    Dataset,
    GroupedData,
    GroupedDataset,
    GroupedSlicedDataset,
    SlicedData,
    SlicedDataset,
)

DATASET_TYPES = [
    Dataset,
    SlicedDataset,
    GroupedDataset,
    AnalyzedDataset,
    GroupedSlicedDataset,
    AnalyzedGroupedDataset,
    AnalyzedSlicedDataset,
    AnalyzedGroupedSlicedDataset,
]

DATA_TYPES = [SlicedData, GroupedData, AnalyzedData]


def typestring2type(type_str: str) -> Any:
    """Turn the name of a class from the module .data into the corresponding type."""
    return getattr(data_module, type_str)


def create_dataset_of_type(T):
    """Since the subtype of Dataset cannot be instantiated out of thin air,
    generate them from supertypes recursively."""
    if T == Dataset:
        return Dataset()
    super_type = typestring2type(T.assert_types[0])
    super_object = create_dataset_of_type(super_type)
    return T(super_object)


@lru_cache()
def can_convert(convert_type, input_type) -> bool:
    if convert_type == Dataset:
        return True
    assert_types = tuple(typestring2type(t) for t in convert_type.assert_types)
    if not issubclass(input_type, assert_types):
        return False
    excluded_types = tuple(typestring2type(t) for t in convert_type.excluded_types)
    return not issubclass(input_type, excluded_types)


@pytest.mark.parametrize("input_type", DATASET_TYPES)
@pytest.mark.parametrize("conversion_type", DATASET_TYPES)
def test_dataset2dataset(input_type, conversion_type):
    i_name, c_name = input_type.__name__, conversion_type.__name__
    input_object = create_dataset_of_type(input_type)
    if can_convert(conversion_type, input_type):
        converted = conversion_type(input_object)
        print(f"\nConverted {c_name}({i_name}) -> {type(converted).__name__}")
        assert isinstance(converted, conversion_type)
        if isinstance(converted, GroupedData):
            assert hasattr(converted, "grouped_indices")
    else:
        try:
            conversion_type(input_object)
            print(
                f"This should not have been printed because converting {i_name} -> {c_name} is excluded."
            )
            assert False
        except TypeError:
            print(f"Converting {i_name} -> {c_name} failed as expected.")


@pytest.mark.parametrize("input_type", DATASET_TYPES)
@pytest.mark.parametrize("conversion_type", DATA_TYPES)
def test_dataset_type(input_type, conversion_type):
    i_name, c_name = input_type.__name__, conversion_type.__name__
    input_object = create_dataset_of_type(input_type)
    if can_convert(conversion_type, input_type):
        converted = conversion_type(input_object)
        print(f"\nConverted {c_name}({i_name}) -> {type(converted).__name__}")
        assert isinstance(converted, input_type)
        assert isinstance(converted, conversion_type)
        if isinstance(converted, GroupedData):
            assert hasattr(converted, "grouped_indices")
    else:
        try:
            conversion_type(input_object)
            print(
                f"This should not have been printed because converting {i_name} -> {c_name} is excluded."
            )
            assert False
        except TypeError:
            print(f"Converting {i_name} -> {c_name} failed as expected.")
