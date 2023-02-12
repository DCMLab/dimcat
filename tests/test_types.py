from functools import lru_cache, partial
from itertools import cycle, islice
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import dimcat.data as data_module
import numpy as np
import pandas as pd
import pytest
from dimcat._typing import PieceID, TypedSequence, to_tuple
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
from ms3 import scale_degree2name
from pandas._testing import assert_frame_equal

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
        assert hasattr(converted, "indices")
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


@pytest.fixture(
    scope="class",
    params=["fromList", "fromTuple", "fromSeries", "fromArray", "fromIter", "fromDict"],
)
def iterable_type(request):
    return request.param


def to_list(elem):
    return [elem]


def to_piece_id(
    elem: Union[
        str,
        Iterable,
    ]
) -> PieceID:
    if isinstance(elem, PieceID):
        return elem
    if isinstance(elem, str):
        result = PieceID(elem, "piece")
    elif isinstance(elem, Iterable):
        corpus, name = (str(e) for e in islice(cycle(elem), 2))
        result = PieceID(corpus, name)
    else:
        result = PieceID(str(elem), "piece")
    return result


fifths2C_minor = partial(scale_degree2name, localkey="bv", globalkey="f#")

# test_id -> (values, converter, expected_type)
test_ids: Dict[str, Tuple] = {
    "float->float": ([1.0, 2.0, 3.0], None, float),
    "int->int": ([1, 2, 3], None, (int, np.int64)),
    "str->str": (["1", "two", "three"], None, (str, np.str_)),
    "mixed->tuple": ([("1",), "2", "3"], None, tuple),
    "mixed->Error": ([1, "two", "three"], None, TypeError),
    "str->int": (["1", "2", "3"], int, int),
    "mixed->str": ([1, "two", 3.0], str, str),
    "list(mixed)->list": ([1, "two", 3.0], to_list, list),
    "tuple(mixed)->tuple": ([1, "two", 3.0], to_tuple, tuple),
    "PieceID(tuple)->PieceID": ([("C1", "P1"), ("C1", "P2")], PieceID._make, PieceID),
    "str(bass)->str": (
        [
            0,
            2,
            -3,
            0,
            -1,
            1,
            -4,
            -1,
            5,
            1,
            0,
            -2,
            3,
            -4,
            1,
            1,
            6,
            -1,
            -3,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
        ],
        fifths2C_minor,
        str,
    ),
}


@pytest.fixture(scope="class", ids=list(test_ids.keys()), params=test_ids.values())
def typed_sequence_params(request) -> Tuple[List, Optional[Callable], Type]:
    return request.param


class TestTypedSequence:
    # class attributes are set by the .setup() method
    TS: TypedSequence
    expected_type: Type

    @pytest.fixture(autouse=True, scope="class")
    def setup(self, iterable_type, typed_sequence_params):
        """Instantiate the TypedSequence for testing based on the parameters."""
        values, converter, expected_type = typed_sequence_params
        match iterable_type:
            case "fromList":
                values: List = list(values)
            case "fromTuple":
                values: Tuple = tuple(values)
            case "fromSeries":
                values: pd.Series = pd.Series(values)
            case "fromArray":
                values: np.ndarray = np.array(values, dtype="object")
            case "fromIter":
                values: Iterator = iter(values)
                expected_type = TypeError
            case "fromDict":
                values: Dict = dict.fromkeys(values)
                expected_type = TypeError
        if not isinstance(expected_type, tuple) and issubclass(
            expected_type, BaseException
        ):
            print(
                f"Should raise {expected_type} with values={values}, converter={converter}"
            )
            with pytest.raises(expected_type):
                _ = TypedSequence(values=values, converter=converter)
            pytest.skip(f"Correctly raised {expected_type}")
        TestTypedSequence.TS = TypedSequence(values=values, converter=converter)
        TestTypedSequence.expected_type = expected_type

    def test_types(self):
        print(self.TS)
        for elem in self.TS:
            if not isinstance(elem, self.expected_type):
                print(type(elem))
                assert isinstance(elem, self.expected_type)

    @pytest.fixture(
        ids=[
            "append_int",
            "append_str",
            "append_float",
            "append_tuple",
            "append_PieceID",
        ],
        params=[1, "one", 1.7, ("C1", "P2"), PieceID("C1", "P3")],
    )
    def new_element(self, request):
        return request.param

    def test_append(self, new_element):
        print(f"Before: {self.TS}")
        if not isinstance(new_element, self.TS.dtype):
            print(f"Should raise TypeError when appending {new_element}")
            with pytest.raises(TypeError):
                self.TS.append(new_element)
            return
        before = len(self.TS)
        self.TS.append(new_element)
        after = len(self.TS)
        print(f"After: {self.TS}")
        assert after == before + 1

    @pytest.fixture(
        ids=["map_str", "map_list", "map_int", "map_piece_id"],
        params=[(str, str), (to_list, list), (int, int), (to_piece_id, PieceID)],
    )
    def func_and_result_type(self, request):
        return request.param

    def test_map(self, func_and_result_type):
        func, result_type = func_and_result_type
        print(f"TypedSequence before mapping {func}: {self.TS}")
        if func is int and self.TS.dtype in (list, tuple, PieceID, str):
            print(f"Should raise TypeError when mapping {func} onto {self.TS.name}")
            with pytest.raises(TypeError):
                _ = self.TS.map(func)
            return
        sequence_map = self.TS.map(func)
        print(f"After: {sequence_map}")
        native_map = list(map(func, self.TS))
        assert sequence_map.dtype == result_type
        assert sequence_map == native_map

    def test_sequence_of_sequences(self):
        """Creates a sequence containing itself and a copy containing twice its elements."""
        as_list = list(self.TS) * 2
        s_of_s = TypedSequence([self.TS, as_list])
        first, second = s_of_s
        assert second.dtype == first.dtype
        assert len(second) == len(first) * 2

    def test_bigrams(self):
        bigrams = self.TS.get_n_grams(2)
        print(f"Bigrams of {self.TS}:\n{bigrams}")
        if self.TS.dtype == list:
            # lists will be treated as sequences themselves and since they have length 1, won't yield any bigrams
            return
        assert bigrams.dtype == tuple
        tm_sequence = self.TS.get_transition_matrix(2)
        tm_bigrams = bigrams.get_transition_matrix(2)
        assert_frame_equal(tm_sequence, tm_bigrams)
        print(tm_sequence)
        tm_sequence = self.TS.get_transition_matrix(2, sort=True)
        tm_bigrams = bigrams.get_transition_matrix(2, sort=True)
        assert_frame_equal(tm_sequence, tm_bigrams)
