from __future__ import annotations

import logging
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Hashable,
    Iterable,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    TypeAlias,
    TypeVar,
    Union,
    _GenericAlias,
    get_args,
    overload,
    runtime_checkable,
)

import ms3
import numpy as np
import pandas as pd
from dimcat.utils import grams, transition_matrix
from scipy.stats import entropy

logger = logging.getLogger(__name__)


class PieceID(NamedTuple):
    corpus: str
    piece: str


Pandas: TypeAlias = Union[pd.Series, pd.DataFrame]
GroupID: TypeAlias = tuple
SliceID: TypeAlias = Tuple[str, str, pd.Interval]
ID: TypeAlias = Union[PieceID, SliceID]
T_co = TypeVar("T_co", covariant=True)
T_hash = TypeVar("T_hash", bound=Hashable, covariant=True)
C = TypeVar("C")  # convertible
Out = TypeVar("Out")  # output
TS = TypeVar("TS", bound="TypedSequence")


class TypedSequence(Sequence[T_co]):
    """A TypedSequence behaves like a list in many aspects but with the difference that it
    imposes one particular data type on its elements.

    If it is instantiated without a converter, the type will be inferred from the first element:

        >>> A = TypedSequence([[1], '2', {7}])
        TypedSequence[list]([[1], ['2'], [7]])

    However, this only works if ``type(a)`` yields a constructor, that works on all elements,
    otherwise a converter needs to be passed:

        >>> converter = lambda e: (e,)
        >>> B = TypedSequence([1, 2, 3], converter)
        TypedSequence[tuple]([(1,), (2,), (3,)])

    TypedSequences can be nested, i.e. have other TypedSequences as elements. The base class,
    however, does not enforce equal type parameters on them:

        >>> C = TypedSequence([A, B, [1, 2.1, 3.9]])
        TypedSequence[TypedSequence]([TypedSequence[list]([[1], ['2'], [7]]),
                                      TypedSequence[tuple]([(1,), (2,), (3,)]),
                                      TypedSequence[int]([1, 2, 3])])

    The module as a few example of parametrized subtypes. Notably, subtypes can register
    themselves as default type for instantiating a TypedSequence with a particular first value.
    This is useful for downcasting to a subclass that has the fitting converter pre-defined.
    For example, ``PieceIndex`` is defined as

        class PieceIndex(TypedSequence[PieceID], register_for=[PieceID]):

    where ``register_for=List[Type]`` makes sure any TypedSequence instantiated
    **without a custom converter** and with a first element of type ``Type`` will be
    cast to that subclass.
    """

    _type_parameter: Optional[T_co] = None
    """Stores the value of the parametrized type."""
    _type2subclass: Dict[Type[T_co], Type[TypedSequence]] = {}
    """Registry of all subclasses that are defined with ``register_for=[type,...]``. Whenever a TypedSequence is
    initiated without converter, the __new__() method looks at the type of the first value (if any) and, if it
    is contained in the registry, creates an object from the pertinent subclass.
    """

    def __len__(self):
        return len(self.values)

    def __getitem__(self, int_or_slice):
        if isinstance(int_or_slice, Union[int, slice]):
            return self.values[int_or_slice]
        raise KeyError(f"{self.name} cannot be subscripted with {int_or_slice}")

    def to_series(self):
        try:
            S = pd.Series(self.values, dtype=self.dtype)
        except Exception:
            S = pd.Series(self.values, dtype=object)
        return S

    def map(
        self,
        func: Callable[[T_co], Out],
    ) -> TypedSequence[Out]:
        try:
            values = list(map(func, self.values))
        except Exception as e:
            raise TypeError(f"Mapping {func} onto {self.name} failed with:\n'{e}'")
        sequential = TypedSequence(values=values)
        return sequential

    def filtered_by_condition(
        self,
        condition: Callable[[T_co], bool],
    ) -> Iterator[T_co]:
        yield from (x for x in self.values if condition(x))

    def get_n_grams(self, n: int) -> Ngrams[T_co]:
        """
        Returns n-gram tuples of the sequence, i.e. all N-(n-1) possible direct successions of n elements.
        """
        n_grams = grams(self.values, n=n)
        if n == 2:
            return Bigrams(values=n_grams)
        else:
            return Ngrams(values=n_grams)

    def get_transition_matrix(
        self,
        n=2,
        k=None,
        smooth: int = 0,
        normalize: bool = False,
        entropy: bool = False,
        excluded_symbols: Optional[List] = None,
        distinct_only: bool = False,
        sort: bool = False,
        percent: bool = False,
        decimals: Optional[int] = None,
    ) -> pd.DataFrame:
        """Returns a transition table of n-grams, showing the frequencies with which any subsequence of length n-1
        is followed by any of the n-grams' last elements.

        Args:
            list_of_sequences:
                List of elements or nested list of elements between which the transitions are calculated. If the list
                is nested, bigrams are calculated recursively to exclude transitions between the lists. If you want
                to create the transition matrix from a list of n-grams directly, pass it as ``list_of_grams`` instead.
            list_of_grams: List of tuples being n-grams. If you want to have them computed from a list of sequences,
                pass it as ``list_of_sequences`` instead.
            n: If ``list_of_sequences`` is passed, the number of elements per n-gram tuple. Ignored otherwise.
            k: If specified, the transition matrix will show only the top k n-grams.
            smooth: If specified, this is the minimum value in the transition matrix.
            normalize: By default, absolute counts are shown. Pass True to normalize each row.
            entropy: Pass True to add a column showing the normalized entropy for each row.
            excluded_symbols: Any n-gram containing any of these symbols will be excluded.
            distinct_only: Pass True to exclude all n-grams consisting of identical elements only.
            sort:
                By default, the columns are ordered by n-gram frequency.
                Pass True to sort them separately, i.e. each by their own frequencies.
            percent: Pass True to multiply the matrix by 100 before rounding to `decimals`
            decimals: To how many decimals you want to round the matrix, if at all.

        Returns:
            DataFrame with frequency statistics of (n-1) grams transitioning to all occurring last elements.
            The index is made up of strings corresponding to all but the last element of the n-grams,
            with the column index containing all last elements.
        """
        return transition_matrix(
            list_of_sequences=self.values,
            n=n,
            k=k,
            smooth=smooth,
            normalize=normalize,
            entropy=entropy,
            excluded_symbols=excluded_symbols,
            distinct_only=distinct_only,
            sort=sort,
            percent=percent,
            decimals=decimals,
        )

    def __init_subclass__(cls, register_for=None, **kwargs):
        """Registers every subclass under the class variable :attr:`_registry`"""
        super().__init_subclass__(**kwargs)
        if register_for is not None:
            for dtype in register_for:
                if dtype in cls._type2subclass:
                    raise KeyError(
                        f"Type {dtype} had already been registered by {cls._type2subclass[dtype]}."
                    )
                cls._type2subclass[dtype] = cls
            logger.debug(
                f"{cls}: TypedSequence will default to a [{cls}] if first value is {register_for}."
            )
        # The following two lines make the value of T available for all parametrized subclasses.
        # Thanks to Paweł Rubin via https://stackoverflow.com/a/71720366
        cls._type_parameter = get_args(cls.__orig_bases__[0])[0]
        logger.debug(f"{cls}._type_parameter = {get_args(cls.__orig_bases__[0])[0]}")

    def __new__(
        cls,
        values: Sequence[Union[T_co, C]],
        converter: Optional[Callable[[C], T_co]] = None,
        **kwargs,
    ):
        if not isinstance(values, (Sequence, np.ndarray, pd.Series)):
            raise TypeError(
                f"The first argument needs to be a Sequence, not {type(values)}."
            )
        nonempty = len(values) > 0
        if converter is None and nonempty:
            first_type = type(values[0])
            if first_type in TypedSequence._type2subclass:
                new_object_type = TypedSequence._type2subclass[first_type]
                logger.debug(
                    f"Creating {new_object_type} because {first_type} is in {TypedSequence._type2subclass.keys()}"
                )
                return super().__new__(new_object_type)
            logger.debug(
                f"Creating {cls} because {first_type} is not in {TypedSequence._type2subclass.keys()}"
            )
        return super().__new__(cls)

    @overload
    def __init__(self, values: Sequence[T_co], converter: Literal[None]):
        ...

    @overload
    def __init__(self, values: Sequence[C], converter: Callable[[C], T_co]):
        ...

    def __init__(
        self,
        values: Sequence[Union[T_co, C]],
        converter: Optional[Callable[[C], T_co]] = None,
        **kwargs,
    ):
        """Sequence object that converts all elements to the same data type.

        If no converter is passed, it is inferred in two different ways:
        a) If self is a subclass of TypedSequence that has been parametrized with class T,
           T is used as a converter. That is, it needs to work as a constructor, which is
           tested using callable(T). Otherwise:
        b) As a fallback, the type of the first value is used for the entire sequence.

        Args:
            values:
                The values you want to create the sequence from. If one of them cannot be converted,
                a TypeError will be thrown.
            converter: A callable that converts values of all expected/allowed types to T_co.
        """
        logger.debug(
            f"{self.__class__.__name__}(values={list(values)}, converter={converter})"
        )
        self._values: List[T_co] = []
        self._converter: Optional[Callable[[C], T_co]] = None
        self.converter = converter
        self.values = values

    @property
    def converter(self) -> Optional[Callable[[Any], T_co]]:
        if self._converter is not None:
            return self._converter
        if not isinstance(self._type_parameter, (TypeVar, _GenericAlias)):
            return self._type_parameter

    @converter.setter
    def converter(self, converter: Optional[Callable[[Any], T_co]]):
        self._converter = converter

    @property
    def dtype(self) -> Optional[Type]:
        if self._type_parameter is None or isinstance(
            self._type_parameter, (TypeVar, _GenericAlias)
        ):
            if len(self.values) > 0:
                return type(self.values[0])
            else:
                return self.converter
        return self._type_parameter

    @property
    def name(self) -> str:
        name = self.__class__.__name__
        if self.dtype is None:
            name += "[None]"
        else:
            name += f"[{self.dtype.__name__}]"
        return name

    @property
    def values(self) -> List[T_co]:
        return list(self._values)

    @values.setter
    def values(self, values: Sequence[Union[T_co, C]]):
        self._values = [self.convert(val) for val in values]

    def convert(self, value: Union[T_co, C]) -> T_co:
        if self.converter is None:
            # this should happen only once in the object's lifetime
            if all(
                (
                    self._type_parameter is not None,
                    callable(self._type_parameter),
                    not isinstance(self._type_parameter, TypeVar),
                )
            ):
                self.converter = self._type_parameter
            else:
                self.converter = type(value)
        try:
            return self.converter(value)
        except Exception as e:
            raise TypeError(
                f"Conversion {self.converter.__name__}({value}) -> {self.dtype} failed with:\n'{e}'"
            )

    @overload
    def append(self, value: T_co, convert: Literal[False]) -> None:
        ...

    @overload
    def append(self, value: Union[T_co, C], convert: Literal[True]) -> None:
        ...

    def append(self, value: Union[T_co, C], convert: bool = False) -> None:
        if convert:
            self._values.append(self.convert(value))
        else:
            try:
                type_check = isinstance(value, self.dtype)
            except Exception as e:
                if len(self._values) == 0:
                    try:
                        converted_value = self.convert(value)
                    except Exception:
                        raise ValueError(
                            f"The exact dtype of this empty sequence is not yet defined and "
                            f"{value} cannot be converted with {self.converter}."
                        )
                    value_type, converted_type = type(value), type(converted_value)
                    if issubclass(value_type, converted_type):
                        logger.debug(
                            f"First value {value} of type {value_type} is compatible with the type "
                            f"yielded by the converter {self.converter}."
                        )
                        self._values.append(value)
                    else:
                        raise TypeError(
                            f"This sequence is empty but the first value to be appended, {value} "
                            f"has type {value_type}, which seems to be incompatible with the type "
                            f"{converted_type} yielded by the converter {self.converter}. Try setting "
                            f"convert=True."
                        )
                else:
                    raise TypeError(
                        f"Checking the type of {value} against {self.dtype} failed with\n{e}"
                    )
            if type_check:
                self._values.append(value)
            else:
                raise TypeError(
                    f"Cannot append {value} to {self.name}. Try setting convert=True."
                )

    @overload
    def extend(self, values: Iterable[T_co], convert: Literal[False]) -> None:
        ...

    @overload
    def extend(self, values: Iterable[Union[T_co, C]], convert: Literal[True]) -> None:
        ...

    def extend(self, values: Iterable[Union[T_co, C]], convert: bool = False) -> None:
        for value in values:
            self.append(value=value, convert=convert)

    def unique(self) -> TypedSequence[T_co]:
        unique_values = (
            self.to_series().unique()
        )  # keeps order of first occurrence, unlike using set()
        return TypedSequence(unique_values)

    def get_changes(self) -> TypedSequence[T_co]:
        """Transforms values [A, A, A, B, C, C, A, C, C, C] --->  [A, B, C, A, C]"""
        prev = object()
        occurrence_list = [
            prev := v for v in self.to_series() if prev != v  # noqa: F841
        ]
        return TypedSequence(occurrence_list)

    def count(self) -> pd.Series:
        """Count the occurrences of objects in the sequence"""
        return self.to_series().value_counts()

    def mean(self) -> float:
        return self.to_series().mean()

    def probability(self) -> pd.Series:
        return self.to_series().value_counts(normalize=True)

    def entropy(self) -> float:
        """
        The Shannon entropy (information entropy), the expected/average surprisal based on its probability distrib.
        """
        # mean_entropy = self.event_entropy().mean()
        p = self.probability()
        distr_entropy = entropy(p, base=2)
        return distr_entropy

    def surprisal(self) -> pd.Series:
        """The self entropy, information content, surprisal"""
        probs = self.probability()
        self_entropy = -np.log(probs)
        series = pd.Series(data=self_entropy, name="surprisal")
        return series

    def __eq__(self, other) -> bool:
        """Considered as equal when 'other' is a Sequence containing the same values."""
        if isinstance(other, Sequence):
            return all(a == b for a, b in zip(self.values, other))
        return False

    def __repr__(self):
        return f"{self.name}({self.values})"

    def __str__(self):
        return f"{self.name}({self.values})"


def to_tuple(elem: Union[T_co, Iterable[T_co]]) -> Tuple[T_co]:
    if isinstance(elem, str):
        return (elem,)
    if isinstance(elem, Iterable):
        return tuple(elem)
    return (elem,)


class Ngrams(TypedSequence[Tuple[T_co, ...]]):
    """N-grams know that they do not need to be converted to n-grams to compute
    a transition matrix.
    """

    def __init__(self, values: Sequence[Sequence[T_co]], converter=to_tuple):
        super().__init__(values, converter)

    def get_transition_matrix(
        self,
        n=2,
        k=None,
        smooth: int = 0,
        normalize: bool = False,
        entropy: bool = False,
        excluded_symbols: Optional[List] = None,
        distinct_only: bool = False,
        sort: bool = False,
        percent: bool = False,
        decimals: Optional[int] = None,
    ) -> pd.DataFrame:
        """Returns a transition table of n-grams, showing the frequencies with which any subsequence of length n-1
        is followed by any of the n-grams' last elements.

        Args:
            n: If ``list_of_sequences`` is passed, the number of elements per n-gram tuple. Ignored otherwise.
            k: If specified, the transition matrix will show only the top k n-grams.
            smooth: If specified, this is the minimum value in the transition matrix.
            normalize: By default, absolute counts are shown. Pass True to normalize each row.
            entropy: Pass True to add a column showing the normalized entropy for each row.
            excluded_symbols: Any n-gram containing any of these symbols will be excluded.
            distinct_only: Pass True to exclude all n-grams consisting of identical elements only.
            sort:
                By default, the columns are ordered by n-gram frequency.
                Pass True to sort them separately, i.e. each by their own frequencies.
            percent: Pass True to multiply the matrix by 100 before rounding to `decimals`
            decimals: To how many decimals you want to round the matrix, if at all.

        Returns:
            DataFrame with frequency statistics of (n-1) grams transitioning to all occurring last elements.
            The index is made up of strings corresponding to all but the last element of the n-grams,
            with the column index containing all last elements.
        """
        return transition_matrix(
            list_of_grams=self.values,
            n=n,
            k=k,
            smooth=smooth,
            normalize=normalize,
            entropy=entropy,
            excluded_symbols=excluded_symbols,
            distinct_only=distinct_only,
            sort=sort,
            percent=percent,
            decimals=decimals,
        )


class Bigrams(Ngrams[Tuple[T_co, T_co]]):
    """ToDo: Need to enforce that tuples are indeed pairs."""

    pass


class PieceIndex(TypedSequence[PieceID], register_for=[PieceID]):
    def __init__(self, values: Sequence[Tuple[str, str]], converter=PieceID._make):
        super().__init__(values, converter)


PathLike: TypeAlias = Union[str, Path]


@runtime_checkable
class PLoader(Protocol):
    def __init__(self, directory: Union[PathLike, Collection[PathLike]]):
        pass

    def iter_pieces(self) -> Iterator[Tuple[PieceID, PPiece]]:
        ...


class FacetName(Enum):
    MEASURES = ("measures",)
    NOTES = ("notes",)
    RESTS = ("rests",)
    NOTES_AND_RESTS = ("notes_and_rests",)
    LABELS = ("labels",)
    EXPANDED = ("expanded",)
    FORM_LABELS = ("form_labels",)
    CADENCES = ("cadences",)
    EVENTS = ("events",)
    CHORDS = ("chords",)


class PFacet(Protocol):
    pass


@runtime_checkable
class PPiece(Protocol):
    def get_facet(self, facet=FacetName) -> PFacet:
        ...


@dataclass(frozen=True)
class TabularData(ABC):
    """Wrapper around a :obj:`pandas.DataFrame`."""

    df: pd.DataFrame

    @classmethod
    def from_df(cls, df: pd.DataFrame):
        """Subclasses can implement transformational logic."""
        instance = cls(df=df)
        return instance

    def get_aspect(self, key: str) -> TypedSequence:
        """In its basic form, get one of the columns as a :obj:`TypedSequence`.
        Subclasses may offer additional aspects, such as transformed columns or subsets of the table.
        """
        series: pd.Series = self.df[key]
        sequential_data = TypedSequence(series)
        return sequential_data

    def __getattr__(self, item):
        """Enable using TabularData like a DataFrame."""
        return getattr(self.df, item)


class Facet(TabularData):
    """Classes implementing the PFacet protocol."""

    pass


@dataclass(frozen=True)
class HarmonyInfo(TabularData):
    def modulation_bigrams_list(self) -> List[str]:
        """Returns a list of str representing the modulation bigram. e.g., "f#_IV/V_bIII/V" """
        globalkey = self.df["globalkey"][0]
        localkey_list = self.get_aspect(key="localkey").get_changes()
        mod_bigrams = localkey_list.get_n_grams(n=2)
        mod_bigrams = ["_".join([item[0], item[1]]) for item in mod_bigrams]
        bigrams = [globalkey + "_" + item for item in mod_bigrams]
        return bigrams

    def get_chord_bigrams(self) -> Bigrams:
        chords = self.get_aspect("chord")
        return chords.get_n_grams(2)


@dataclass(frozen=True)
class MeasureInfo(TabularData):
    pass


@dataclass(frozen=True)
class NoteInfo(TabularData):
    @cached_property
    def tpc(self) -> TypedSequence:
        series = self.df["tpc"]
        sequential = TypedSequence.from_series(series=series)
        return sequential


if __name__ == "__main__":
    df = ms3.load_tsv("~/corelli/metadata.tsv")
    t = Facet(df)
    print(t.df)
