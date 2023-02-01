"""Class hierarchy for data types."""
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
from typing import (
    Any,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    overload,
)

import ms3
import pandas as pd
from ms3._typing import ScoreFacet

from ._typing import ID, GroupID, Pandas, PieceID
from .utils import clean_index_levels

logger = logging.getLogger(__name__)

PROCESSED_DATA_FIELDS: Dict[str, Tuple[str, ...]] = {
    "SlicedData": (
        "sliced",
        "slice_info",
    ),
    "GroupedData": ("grouped_indices",),
    "AnalyzedData": ("processed",),
}
"""Name of the data fields that the three types of processing add to any _Dataset object.
Important for copying objects."""


class Data(ABC):
    """
    Subclasses are the dtypes that this library uses. Every PipelineStep accepts one or several
    dtypes.

    The initializer can set parameters influencing how the contained data will look and is able
    to create an object from an existing Data object to enable type conversion.
    """


class _Dataset(Data, ABC):
    def __init__(self, data: Optional[Data] = None, **kwargs):
        """Create a new :obj:`Data` object."""
        self._data = None
        """Protected attribute for storing and internally accessing the loaded data."""

        self.pieces: Dict[ID, Any] = {}
        """{(corpus, fname) -> Any}
        References to the individual pieces contained in the data. The exact type depends on the type of data.
        """

        self.indices: List[ID] = []
        """List of indices (IDs) which serve for accessing individual pieces of data and
        associated metadata. An index is a ('corpus_name', 'piece_name') tuple ("ID") that can have a third element
        identifying a segment/chunk of a piece."""

        self.pipeline_steps = []
        """The sequence of applied PipelineSteps that has led to the current state in reverse
        order (first element was applied last)."""

        self.index_levels = {
            "indices": ["corpus", "fname"],
            "slicer": [],
            "groups": [],
            "processed": [],
        }
        """Keeps track of index level names. Also used for automatic naming of output files."""

        self.group2pandas = None  # ToDo: deprecate this by using different datatypes for different types of results

        # __init__() methods of subclasses should end with:
        # self.data = data

    @property
    @abstractmethod
    def data(self):
        """Get the data field in its raw form."""
        return self._data

    @data.setter
    @abstractmethod
    def data(self, data_object):
        """Implement setting the _data field after performing type check."""
        if data_object is not None:
            raise NotImplementedError
        self._data = data_object

    @property
    def n_indices(self):
        return len(self.indices)

    @abstractmethod
    def copy(self):
        return self.__class__()

    @abstractmethod
    def iter_facet(self, what):
        """Iterate through (potentially grouped) pieces of data."""
        for index in self.indices:
            yield self.get_item(index, what)

    @abstractmethod
    def get_item(self, index, what):
        """Get an individual piece of data."""

    def track_pipeline(
        self,
        pipeline_step,
        group2pandas=None,
        indices=None,
        processed=None,
        grouper=None,
        slicer=None,
    ):
        """Keep track of the applied pipeline_steps and update index level names and group2pandas
        conversion method.

        Parameters
        ----------
        pipeline_step : :obj:`PipelineStep`
        group2pandas : :obj:`str`, optional
        indices : :obj:`str`, optional
        processed : :obj:`str`, optional
        grouper : :obj:`str`, optional
        slicer : :obj:`str`, optional
        """
        self.pipeline_steps = [pipeline_step] + self.pipeline_steps
        if processed is not None:
            if isinstance(processed, str):
                processed = [processed]
            self.index_levels["processed"] = processed
        if indices is not None:
            if indices == "IDs":
                # once_per_group == True
                self.index_levels["indices"] = ["IDs"]
            elif len(self.index_levels["indices"]) == 2:
                self.index_levels["indices"] = self.index_levels["indices"] + [indices]
            else:
                self.index_levels["indices"][2] = indices
            assert 1 <= len(self.index_levels["indices"]) < 4
        if group2pandas is not None:
            self.group2pandas = group2pandas
        if grouper is not None:
            self.index_levels["groups"] = self.index_levels["groups"] + [grouper]
        if slicer is not None:
            self.index_levels["slicer"] = [slicer]

    @abstractmethod
    def load(self):
        """Load data into memory."""

    def group_of_values2series(self, group_dict) -> pd.Series:
        """Converts an {ID -> processing_result} dict into a Series."""
        series = pd.Series(group_dict, name=self.index_levels["processed"][0])
        series.index = self._rename_multiindex_levels(
            series.index, self.index_levels["indices"]
        )
        return series

    def group_of_series2series(self, group_dict) -> pd.Series:
        """Converts an {ID -> processing_result} dict into a Series."""
        lengths = [len(S) for S in group_dict.values()]
        if 0 in lengths:
            group_dict = {k: v for k, v in group_dict.items() if len(v) > 0}
            if len(group_dict) == 0:
                logger.info("Group contained only empty Series")
                return pd.Series()
            else:
                n_empty = lengths.count(0)
                logger.info(
                    f"Had to remove {n_empty} empty Series before concatenation."
                )
        if len(group_dict) == 1 and list(group_dict.keys())[0] == "group_ids":
            series = list(group_dict.values())[0]
            series.index = self._rename_multiindex_levels(
                series.index, self.index_levels["processed"]
            )
        else:
            series = pd.concat(group_dict.values(), keys=group_dict.keys())
            series.index = self._rename_multiindex_levels(
                series.index,
                self.index_levels["indices"] + self.index_levels["processed"],
            )
        return series

    def group2dataframe(self, group_dict) -> pd.DataFrame:
        """Converts an {ID -> processing_result} dict into a DataFrame."""
        try:
            df = pd.concat(group_dict.values(), keys=group_dict.keys())
        except (TypeError, ValueError):
            logger.info(group_dict)
            raise
        df.index = self._rename_multiindex_levels(
            df.index, self.index_levels["indices"] + self.index_levels["processed"]
        )
        return df

    def group2dataframe_unstacked(self, group_dict):
        return self.group2dataframe(group_dict).unstack()

    def _rename_multiindex_levels(self, multiindex: pd.MultiIndex, index_level_names):
        """Renames the index levels based on the _.index_levels dict."""
        try:
            n_levels = multiindex.nlevels
            if n_levels == 1:
                return multiindex.rename(index_level_names[0])
            n_names = len(index_level_names)
            if n_names < n_levels:
                levels = list(range(len(index_level_names)))
                # The level parameter makes sure that, when n names are given, only the first n levels are being
                # renamed. However, this will lead to unexpected behaviour if index levels are named by an integer
                # that does not correspond to the position of another index level, e.g. ('level0_name', 0, 1)
                return multiindex.rename(index_level_names, level=levels)
            elif n_names > n_levels:
                return multiindex.rename(index_level_names[:n_levels])
            return multiindex.rename(index_level_names)
        except (TypeError, ValueError) as e:
            logger.info(
                f"Failed to rename MultiIndex levels {multiindex.names} to {index_level_names}: '{e}'"
            )
            logger.info(multiindex[:10])
            logger.info(f"self.index_levels: {self.index_levels}")
        # TODO: This method should include a call to clean_multiindex_levels and make use of self.index_levels
        return multiindex


class Dataset(_Dataset):
    """An object that represents one or several corpora issued by the DCML corpus initiative.
    Essentially a wrapper for a ms3.Parse object."""

    def __init__(self, data: Optional[Union["Dataset", ms3.Parse]] = None, **kwargs):
        """

        Parameters
        ----------
        data : Data
            Convert a given Data object into a Corpus if possible.
        kwargs
            All keyword arguments are passed to load().
        """
        super().__init__()
        self.pieces: Dict[ID, ms3.Piece] = {}
        """{(corpus, fname) -> :obj:`ms3.Piece`}
        References to the individual :obj:`ms3.Piece` objects contained in this Dataset. They give access to all data
        and metadata available for one piece.
        """
        self.data = data
        if len(kwargs) > 0:
            self.load(**kwargs)

    @property
    def data(self) -> ms3.Parse:
        """Get the data field in its raw form."""
        return self._data

    @data.setter
    def data(self, data_object: Optional[Union["Dataset", ms3.Parse]]) -> None:
        """Check if the assigned object is suitable for conversion."""
        if data_object is None:
            self._data = ms3.Parse()
            return
        is_dataset_object = isinstance(data_object, Dataset)
        is_parse_object = isinstance(data_object, ms3.Parse)
        if not (is_dataset_object or is_parse_object):
            raise TypeError(
                f"{data_object.__class__} could not be converted to a DCML dataset."
            )
        if is_parse_object:
            self._data = data_object
            return
        # else: got a Dataset object and need to copy its fields
        self._data = data_object._data
        self.pieces = dict(data_object.pieces)
        # ^^^ doesn't copy the ms3.Parse and ms3.Piece objects, only the references ^^^
        # vvv all other fields are deepcopied vvv
        if isinstance(data_object, SlicedData) and not isinstance(self, SlicedData):
            self.indices = sorted(set(ID[:2] for ID in data_object.indices))
        elif self.__class__.__name__ == "Dataset":
            self.indices = []
            self.get_indices()
        else:
            self.indices = list(data_object.indices)
        self.index_levels = deepcopy(data_object.index_levels)
        self.pipeline_steps = list(data_object.pipeline_steps)
        self.group2pandas = data_object.group2pandas
        if self.__class__.__name__ == "Dataset":
            return
        dtypes = typestrings2types(PROCESSED_DATA_FIELDS.keys())
        dtypes = dict(zip(PROCESSED_DATA_FIELDS.keys(), dtypes))
        for processed_type, optional_fields in PROCESSED_DATA_FIELDS.items():
            dtype = dtypes[processed_type]
            if not (isinstance(data_object, dtype) and isinstance(self, dtype)):
                continue
            for field in optional_fields:
                setattr(self, field, deepcopy(getattr(data_object, field)))

    def copy(self):
        return self.__class__(data=self)

    def get_facet(self, what: ScoreFacet, unfold: bool = False) -> pd.DataFrame:
        """Uses _.iter_facet() to collect and concatenate all DataFrames for a particular facet.

        Parameters
        ----------
        what : {'form_labels', 'events', 'expanded', 'notes_and_rests', 'notes', 'labels',
                'cadences', 'chords', 'measures', 'rests'}
            What facet to retrieve.
        unfold : :obj:`bool`, optional
            Pass True if you need repeats to be unfolded.

        Returns
        -------
        :obj:`pandas.DataFrame`
        """
        dfs = {
            idx: df
            for idx, df in self.iter_facet(
                what=what,
                unfold=unfold,
            )
        }
        if len(dfs) == 1:
            return list(dfs.values())[0]
        concatenated_groups = pd.concat(
            dfs.values(), keys=dfs.keys(), names=self.index_levels["groups"]
        )
        return clean_index_levels(concatenated_groups)

    def convert_group2pandas(self, result_dict) -> Union[pd.Series, pd.DataFrame]:
        """Converts the {ID -> processing_result} dict using the method specified in _.group2pandas."""
        converters = {
            "group_of_values2series": self.group_of_values2series,
            "group_of_series2series": self.group_of_series2series,
            "group2dataframe": self.group2dataframe,
            "group2dataframe_unstacked": self.group2dataframe_unstacked,
        }
        converter = converters[self.group2pandas]
        pandas_obj = converter(result_dict)
        return clean_index_levels(pandas_obj)

    def load(
        self,
        directory: Optional[Union[str, List[str]]] = None,
        parse_tsv: bool = True,
        parse_scores: bool = False,
        ms: Optional[str] = None,
    ):
        """
        Load and parse all of the desired raw data and metadata.

        Parameters
        ----------
        directory : str
            The path to all the data to load.
        parse_tsv : :obj:`bool`, optional
            By default, all detected TSV files are parsed and their type is inferred.
            Pass False to prevent parsing TSV files.
        parse_scores : :obj:`bool`, optional
            By default, detected scores files are not parsed to save time.
            Call super().__init__(parse_scores=True) if an analyzer needs
            to access parsed MuseScore XML.
        ms : :obj:`str`, optional
            If you pass the path to your local MuseScore 3 installation, ms3 will attempt to parse
            musicXML, MuseScore 2, and other formats by temporarily converting them. If you're
            using the standard path, you may try 'auto', or 'win' for Windows, 'mac' for MacOS,
            or 'mscore' for Linux. In case you do not pass the 'file_re' and the MuseScore
            executable is detected, all convertible files are automatically selected, otherwise
            only those that can be parsed without conversion.
        """
        if ms is not None:
            self.data.ms = ms
        if directory is not None:
            if isinstance(directory, str):
                directory = [directory]
            for d in directory:
                self.data.add_dir(
                    directory=d,
                )
        if parse_tsv:
            self.data.parse_tsv()
        if parse_scores:
            self.data.parse_scores()
        if self.data.n_parsed_tsvs == 0 and self.data.n_parsed_scores == 0:
            logger.info("No files have been parsed for analysis.")
        else:
            self.get_indices()

    def get_indices(self):
        """Fills self.pieces with metadata and IDs for all loaded data. This resets previously
        applied groupings."""
        self.pieces = {}
        # self.group_labels = {}
        for corpus_name, ms3_corpus in self.data.iter_corpora():
            for fname, piece in ms3_corpus.iter_pieces():
                ID = (corpus_name, fname)
                self.pieces[ID] = piece
        self.set_indices(list(self.pieces.keys()))

    def set_indices(
        self, new_indices: Union[List[PieceID], Dict[Any, List[PieceID]]]
    ) -> None:
        """Replace :attr:`indices` with a new list of IDs.

        Args:
            new_indices:
                The new list IDs or a dictionary of several lists of IDs. The latter is useful for re-grouping
                freshly sliced IDs of a :class:`GroupedDataset`.
        """
        if isinstance(new_indices, dict):
            new_indices = sum(new_indices.values(), [])
        self.indices = new_indices

    def iter_facet(
        self, what: ScoreFacet, unfold: bool = False
    ) -> Iterator[Tuple[ID, pd.DataFrame]]:
        """Iterate through facet DataFrames.

        Args:
            what: Which type of facet to retrieve.
            unfold: Pass True if you need repeats to be unfolded.

        Yields:
            Index tuple.
            Facet DataFrame.
        """
        for index in self.indices:
            df = self.get_item(index, what=what, unfold=unfold)
            if df is None or len(df.index) == 0:
                logger.info(f"{index} has no {what}.")
                continue
            yield index, df

    def get_previous_pipeline_step(self, idx=0, of_type=None):
        """Retrieve one of the previously applied PipelineSteps, either by index or by type.

        Parameters
        ----------
        idx : :obj:`int`, optional
            List index used if ``of_type`` is None. Defaults to 0, which is the PipeLine step
            most recently applied.
        of_type : :obj:`PipelineStep`, optional
            Return the most recently applied PipelineStep of this type.

        Returns
        -------
        :obj:`PipelineStep`
        """
        if of_type is None:
            n_previous_steps = len(self.pipeline_steps)
            try:
                return self.pipeline_steps[idx]
            except IndexError:
                logger.info(
                    f"Invalid index idx={idx} for list of length {n_previous_steps}"
                )
                raise
        try:
            return next(
                step for step in self.pipeline_steps if isinstance(step, of_type)
            )
        except StopIteration:
            raise StopIteration(
                f"Previously applied PipelineSteps do not include any {of_type}: {self.pipeline_steps}"
            )

    @lru_cache()
    def get_item(self, index, what, unfold=False, multiindex=False):
        """Retrieve a DataFrame pertaining to the facet ``what`` of the piece ``index``. If
        the facet has been sliced before, the sliced DataFrame is returned.

        Parameters
        ----------
        index : :obj:`tuple`
            (corpus, fname) or (corpus, fname, interval)
        what : {'form_labels', 'events', 'expanded', 'notes_and_rests', 'notes', 'labels',
                'cadences', 'chords', 'measures', 'rests'}
            What facet to retrieve.
        unfold : :obj:`bool`, optional
            Pass True if you need repeats to be unfolded.
        multiindex : :obj:`bool`, optional
            By default, has one level, which is a :obj:``pandas.IntervalIndex``. Pass True to
            prepend ``index`` as an additional index level.

        Returns
        -------
        :obj:`pandas.DataFrame`
        """
        corpus, fname, *_ = index
        file, df = self.data[corpus][fname].get_facet(
            what, unfold=unfold, interval_index=True
        )
        logger.debug(f"Retrieved {what} from {file}.")
        if df is not None and not isinstance(df.index, pd.IntervalIndex):
            logger.info(f"'{what}' of {index} does not come with an IntervalIndex")
            df = None
        if df is None:
            return
        assert (
            df.index.nlevels == 1
        ), f"Retrieved DataFrame has {df.index.nlevels}, not 1"
        if multiindex:
            df = pd.concat([df], keys=[index[:2]], names=["corpus", "fname"])
        return df


TYPE_CACHE = {}


def typestrings2types(typestrings: Union[str, Collection[str]]) -> Tuple[Any]:
    global TYPE_CACHE
    if isinstance(typestrings, str):
        typestrings = [typestrings]
    result = []
    all_objects = None
    for typ in typestrings:
        if typ not in TYPE_CACHE:
            if all_objects is None:
                all_objects = globals()
            TYPE_CACHE[typ] = all_objects[typ]
        result.append(TYPE_CACHE[typ])
    return tuple(result)


class _ProcessedData(_Dataset):
    """Base class for types of processed :obj:`_Dataset` objects.
    Processed datatypes are created by passing a _Dataset object. The new object will be a copy of the Data with the
    :attr:`prefix` prepended. Subclasses should have an __init__() method that calls super().__init__() and then
    adds additional fields.
    """

    assert_types: Union[str, Collection[str]] = ["Dataset"]
    """Objects raise TypeError upon instantiation if the passed data are not of one of these types."""
    excluded_types: Union[str, Collection[str]] = []
    """Objects raise TypeError upon instantiation if the passed data are of one of these types."""
    type_mapping: Dict[Union[str, Collection[str]], str] = {}
    """{Input type(s) -> Output type}. __new__() picks the first 'value' where the input Data are of type 'key'.
    Objects raise TypeError if nothing matches. object or Data can be used as fallback/default key.
    """

    def __new__(cls, data: Data, **kwargs):
        """Depending on the type of ``data`` (currently only :class:`Dataset` is implemented),
        the new object is turned into the Dataset subtype that corresponds to the performed processing step.

        This method uses the class properties :attr:`assert_types` and :attr:`excluded_types` to determine if the
        input Dataset can actually undergo the current type of processing. Then it uses the class property
        :attr:`type_mapping` to determine the type of the new object to be created.


        Args:
            data: Dataset to be converted into a processed subtype.
            **kwargs:
        """
        assert_types = typestrings2types(cls.assert_types)
        if not isinstance(data, assert_types):
            raise TypeError(
                f"{cls.__name__} objects can only be created from {cls.assert_types}, not '{type(data).__name__}'"
            )
        excluded_types = typestrings2types(cls.excluded_types)
        if isinstance(data, excluded_types):
            raise TypeError(
                f"{cls.__name__} objects cannot be created from '{type(data).__name__}' because it is among the "
                f"excluded_types {cls.excluded_types}."
            )
        type_mapping = {
            typestrings2types(input_type): typestrings2types(output_type)[0]
            for input_type, output_type in cls.type_mapping.items()
        }
        new_obj_type = None
        for input_type, output_type in type_mapping.items():
            if isinstance(data, input_type):
                new_obj_type = output_type
                break
        if new_obj_type is None:
            raise TypeError(
                f"{cls.__name__} no output type defined for '{type(data)}', only for {list(type_mapping.keys())}."
            )
        obj = object.__new__(new_obj_type)
        # obj.__init__(data=data, **kwargs)
        return obj

    def __init__(self, data: Data, **kwargs):
        super().__init__(data=data, **kwargs)


class SlicedData(_ProcessedData):
    """A type of Data object that contains the slicing information created by a Slicer. It slices all requested
    facets based on that information.
    """

    excluded_types = ["AnalyzedData", "SlicedData"]
    type_mapping = {
        "GroupedDataset": "GroupedSlicedDataset",
        "Dataset": "SlicedDataset",
    }

    def __init__(self, data: Data, **kwargs):
        logger.debug(f"{type(self).__name__} -> before {super()}.__init__()")
        super().__init__(data=data, **kwargs)
        logger.debug(f"{type(self).__name__} -> after {super()}.__init__()")
        if not hasattr(self, "sliced"):
            self.sliced = {}
            """Dict for sliced data facets."""
        if not hasattr(self, "slice_info"):
            self.slice_info = {}
            """Dict holding metadata of slices (e.g. the localkey of a segment)."""

    def get_slice(self, index, what):
        if what in self.sliced and index in self.sliced[what]:
            return self.sliced[what][index]

    def get_slice_info(self) -> pd.DataFrame:
        """Concatenates slice_info Series and returns them as a DataFrame."""
        if len(self.slice_info) == 0:
            logger.info("No slices available.")
            return pd.DataFrame()
        concatenated_info = pd.concat(
            self.slice_info.values(), keys=self.slice_info.keys(), axis=1
        ).T
        concatenated_info.index.rename(self.index_levels["indices"], inplace=True)
        return concatenated_info

    # def iter_facet(self, what, unfold=False, concatenate=False, ignore_groups=False):
    #     """Iterate through groups of potentially sliced facet DataFrames.
    #
    #     Parameters
    #     ----------
    #     what : {'form_labels', 'events', 'expanded', 'notes_and_rests', 'notes', 'labels',
    #             'cadences', 'chords', 'measures', 'rests'}
    #         What facet to retrieve.
    #     unfold : :obj:`bool`, optional
    #         Pass True if you need repeats to be unfolded.
    #     concatenate : :obj:`bool`, optional
    #         By default, the returned dict contains one DataFrame per ID in the group.
    #         Pass True to instead concatenate the DataFrames. Then, the dict will contain only
    #         one entry where the key is a tuple containing all IDs and the value is a DataFrame,
    #         the components of which can be distinguished using its MultiIndex.
    #     ignore_groups : :obj:`bool`, False
    #         If set to True, the iteration loop is flattened and yields (index, facet_df) pairs directly. Clashes
    #         with the setting concatenate=True which concatenates facets per group.
    #
    #     Yields
    #     ------
    #     :obj:`tuple`
    #         Group identifier
    #     :obj:`dict` or :obj:`pandas.DataFrame`
    #         Default: {ID -> DataFrame}.
    #         If concatenate=True: DataFrame with MultiIndex identifying ID, and (eventual) interval.
    #     """
    #     if not self.slice_facet_if_necessary(what, unfold):
    #         logger.info(f"No sliced {what} available.")
    #         raise StopIteration
    #     if sum((concatenate, ignore_groups)) > 1:
    #         raise ValueError(
    #             "Arguments 'concatenate' and 'ignore_groups' are in conflict, choose one "
    #             "or use the method get_facet()."
    #         )
    #     for group, index_group in self.iter_groups():
    #         result = {}
    #         missing_id = []
    #         for index in index_group:
    #             df = self.get_item(index, what=what, unfold=unfold)
    #             if df is None:
    #                 continue
    #             elif ignore_groups:
    #                 yield index, df
    #             if len(df.index) == 0:
    #                 missing_id.append(index)
    #             result[index] = df
    #         if ignore_groups:
    #             continue
    #         n_results = len(result)
    #         if len(missing_id) > 0:
    #             if n_results == 0:
    #                 pass
    #                 # logger.info(f"No '{what}' available for {group}.")
    #             else:
    #                 logger.info(
    #                     f"Group {group} is missing '{what}' for the following indices:\n"
    #                     f"{missing_id}"
    #                 )
    #         if n_results == 0:
    #             continue
    #         if concatenate:
    #             if n_results == 1:
    #                 # workaround necessary because of nasty "cannot handle overlapping indices;
    #                 # use IntervalIndex.get_indexer_non_unique" error
    #                 result["empty"] = pd.DataFrame()
    #             result = pd.concat(
    #                 result.values(),
    #                 keys=result.keys(),
    #                 names=self.index_levels["indices"] + ["interval"],
    #             )
    #             result = {tuple(index_group): result}
    #
    #         yield group, result

    def iter_slice_info(self) -> Iterator[Tuple[tuple, pd.DataFrame]]:
        """Iterate through concatenated slice_info Series for each group."""
        for group, index_group in self.iter_grouped_indices():
            group_info = {ix: self.slice_info[ix] for ix in index_group}
            group_df = pd.concat(group_info.values(), keys=group_info.keys(), axis=1).T
            group_df.index = self._rename_multiindex_levels(
                group_df.index, self.index_levels["indices"]
            )
            yield group, group_df

    def slice_facet_if_necessary(self, what, unfold):
        """

        Parameters
        ----------
        what : :obj:`str`
            Facet for which to create slices if necessary
        unfold : :obj:`bool`
            Whether repeats should be unfolded.

        Returns
        -------
        :obj:`bool`
            True if slices are available or not needed, False otherwise.
        """
        if not hasattr(self, "slice_info"):
            # no slicer applied
            return True
        if len(self.slice_info) == 0:
            # applying slicer did not yield any slices
            return True
        if what in self.sliced:
            # already sliced
            return True
        self.sliced[what] = {}
        facet_ids = defaultdict(list)
        for corpus, fname, interval in self.slice_info.keys():
            facet_ids[(corpus, fname)].append(interval)
        for id, intervals in facet_ids.items():
            facet_df = self.get_item(id, what, unfold)
            if facet_df is None or len(facet_df.index) == 0:
                continue
            sliced = ms3.overlapping_chunk_per_interval(facet_df, intervals)
            self.sliced[what].update(
                {id + (iv,): chunk for iv, chunk in sliced.items()}
            )
        if len(self.sliced[what]) == 0:
            del self.sliced[what]
            return False
        return True


class GroupedData(_ProcessedData):
    """A type of Data object that behaves like its predecessor but returns and iterates through groups."""

    type_mapping = {
        (
            "AnalyzedGroupedSlicedDataset",
            "AnalyzedSlicedDataset",
        ): "AnalyzedGroupedSlicedDataset",
        "GroupedSlicedDataset": "GroupedSlicedDataset",
        ("AnalyzedGroupedDataset", "AnalyzedDataset"): "AnalyzedGroupedDataset",
        "SlicedDataset": "GroupedSlicedDataset",
        "Dataset": "GroupedDataset",
    }

    def __init__(self, data: Data, **kwargs):
        logger.debug(f"{type(self).__name__} -> before {super()}.__init__()")
        super().__init__(data=data, **kwargs)
        logger.debug(f"{type(self).__name__} -> after {super()}.__init__()")
        if not hasattr(self, "grouped_indices"):
            if hasattr(data, "grouped_indices"):
                self.grouped_indices = data.grouped_indices
            else:
                self.grouped_indices: Dict[GroupID, List[ID]] = {(): self.indices}
                """{group_key -> indices} dictionary of indices (IDs) which serve for accessing individual pieces of
                data and associated metadata. An index is a ('corpus_name', 'piece_name') tuple ("ID")
                that can have a third element identifying a segment/chunk of a piece.
                The group_keys are an empty tuple by default; with every applied Grouper,
                the length of all group_keys grows by one and the number of group_keys grows or stays the same."""

    def iter_grouped_indices(self):
        """Iterate through groups of indices as defined by the previously applied Groupers.

        Yields
        -------
        :obj:`tuple` of :obj:`str`
            A tuple of keys reflecting the group hierarchy
        :obj:`list` of :obj:`tuple`
            A list of IDs belonging to the same group.
        """
        if len(self.indices) == 0:
            raise ValueError("No data has been loaded.")
        if any(len(index_list) == 0 for index_list in self.grouped_indices.values()):
            logger.warning("Data object contains empty groups.")
        yield from self.grouped_indices.items()


class AnalyzedData(_ProcessedData):
    """A type of Data object that contains the results of an Analyzer and knows how to plot it."""

    type_mapping = {
        (
            "AnalyzedGroupedSlicedDataset",
            "GroupedSlicedDataset",
        ): "AnalyzedGroupedSlicedDataset",
        ("AnalyzedSlicedDataset", "SlicedDataset"): "AnalyzedSlicedDataset",
        ("AnalyzedGroupedDataset", "GroupedDataset"): "AnalyzedGroupedDataset",
        "Dataset": "AnalyzedDataset",
    }

    def __init__(self, data: Data, **kwargs):
        logger.debug(f"{type(self).__name__} -> before {super()}.__init__()")
        super().__init__(data=data, **kwargs)
        logger.debug(f"{type(self).__name__} -> after {super()}.__init__()")
        if not hasattr(self, "processed"):
            if hasattr(data, "processed"):
                self.processed = data.processed
            else:
                self.processed: Dict[GroupID, Union[Dict[ID, Any], List[str]]] = {}
                """Analyzers store there result here. Those that compute one result per item per group
                store {ID -> result} dicts, all others store simply the result for each group. In the first case,
                :attr:`group2pandas` needs to be specified for correctly converting the dict to a pandas object."""

    @overload
    def get(self, as_pandas: bool = Literal[True]) -> Pandas:
        ...

    @overload
    def get(self, as_pandas: bool = Literal[False]) -> Dict[GroupID, Any]:
        ...

    def get(self, as_pandas: bool = True) -> Union[Pandas, Dict[GroupID, Any]]:
        """Collects the results of :meth:`iter` to retrieve all processed data at once.

        Args:
            as_pandas:
                By default, the result is a pandas DataFrame or Series where the first levels
                display group identifiers (if any). Pass False to obtain a nested {group -> group_result}
                dictionary instead.

        Returns:
            The contents of :attr:`processed` in original or adapted form.
        """
        if len(self.processed) == 0:
            logger.info("No data has been processed so far.")
            return
        results = {group: result for group, result in self.iter(as_pandas=as_pandas)}
        if not as_pandas:
            return results
        if self.group2pandas is None:
            return pd.Series(results)
        # default: concatenate to a single pandas object
        if len(results) == 1 and () in results:
            pandas_obj = pd.concat(results.values())
        else:
            try:
                pandas_obj = pd.concat(
                    results.values(),
                    keys=results.keys(),
                    names=self.index_levels["groups"],
                )
            except ValueError:
                logger.info(self.index_levels["groups"])
                logger.info(results.keys())
                raise
        return clean_index_levels(pandas_obj)

    @overload
    def iter(
        self, as_pandas: bool = Literal[False], ignore_groups: bool = Literal[False]
    ) -> Iterator[Tuple[GroupID, Union[Dict[ID, Any], Any]]]:
        ...

    @overload
    def iter(
        self, as_pandas: bool = Literal[True], ignore_groups: bool = Literal[False]
    ) -> Iterator[Tuple[GroupID, Union[Pandas, Any]]]:
        ...

    @overload
    def iter(
        self, as_pandas: bool = Literal[False], ignore_groups: bool = Literal[True]
    ) -> Iterator[Union[Tuple[ID, Any], Any]]:
        ...

    @overload
    def iter(
        self, as_pandas: bool = Literal[True], ignore_groups: bool = Literal[True]
    ) -> Iterator[Union[Pandas, Any]]:
        ...

    def iter(
        self, as_pandas: bool = True, ignore_groups: bool = False
    ) -> Iterator[
        Union[
            Tuple[GroupID, Union[Dict[ID, Any], Any]],
            Tuple[GroupID, Union[Pandas, Any]],
            Union[Tuple[ID, Any], Any],
            Union[Pandas, Any],
        ]
    ]:
        """Iterate through :attr:`processed` data.

        Args:
            as_pandas:
                Setting this value to False corresponds to iterating through .processed.items(),
                where keys are group IDs and values are results for Analyzers that compute
                one result per group, or {ID -> result} dicts for Analyzers that compute
                one result per item per group. The default value (True) has no effect in the first case,
                but in the second case, the dictionary will be converted to a Series if the conversion method is
                set in :attr:`group2pandas`.
            ignore_groups:
                If set to True, the iteration loop is flattened and does not include group identifiers. If as_pandas
                is False (default), and the applied Analyzer computes one {ID -> result} dict per group,
                this will correspond to iterating through the (ID, result) tuples for all groups.

        Yields:
            The result of the last applied Analyzer for each group or for each item of each group.
        """
        if ignore_groups and not as_pandas:
            raise ValueError(
                "If you set 'as_dict' and 'ignore_groups' are in conflict, choose one or use _.get()."
            )
        for group, result in self.processed.items():
            if ignore_groups:
                if self.group2pandas is None:
                    yield result
                elif as_pandas:
                    yield self.convert_group2pandas(result)
                else:
                    yield from result.items()
            else:
                if as_pandas and self.group2pandas is not None:
                    yield group, self.convert_group2pandas(result)
                else:
                    yield group, result


class SlicedDataset(SlicedData, Dataset):
    pass


class GroupedDataset(GroupedData, Dataset):
    def iter_grouped_facet(
        self,
        what: ScoreFacet,
    ) -> Iterator[Tuple[GroupID, pd.DataFrame]]:
        """Iterate through one concatenated facet DataFrame per group.

        Args:
            what: Which type of facet to retrieve.

        Yields:
            Group index.
            Facet DataFrame.
        """
        for group, index_group in self.iter_grouped_indices():
            result = {}
            missing_id = []
            for index in index_group:
                df = self.get_item(index, what=what)
                if df is None or len(df.index) == 0:
                    missing_id.append(index)
                    continue
                result[index] = df
            n_results = len(result)
            if len(missing_id) > 0:
                if n_results == 0:
                    pass
                    # logger.info(f"No '{what}' available for {group}.")
                else:
                    logger.info(
                        f"Group {group} is missing '{what}' for the following indices:\n{missing_id}"
                    )
            if n_results == 0:
                continue
            if n_results == 1:
                # workaround necessary because of nasty "cannot handle overlapping indices;
                # use IntervalIndex.get_indexer_non_unique" error
                result["empty"] = pd.DataFrame()
            result = pd.concat(
                result.values(),
                keys=result.keys(),
                names=self.index_levels["indices"] + ["interval"],
            )
            yield group, result

    def set_indices(
        self, new_indices: Union[List[PieceID], Dict[ID, List[PieceID]]]
    ) -> None:
        """Replace :attr:`indices` with a new list of IDs and update the :attr:`grouped_indices` accordingly.

        Args:
            new_indices:
                The new list of IDs or an {old_id -> [new_id]} dictionary to replace the IDs with a list of new IDs.
        """
        id2group = defaultdict(lambda: ())
        if len(self.indices) > 0:
            id2group.update(
                {
                    ID: group
                    for group, group_ids in self.iter_grouped_indices()
                    for ID in group_ids
                }
            )
        new_grouped_indices = defaultdict(list)
        if isinstance(new_indices, dict):
            for old_id, new_ids in new_indices.items():
                old_group = id2group[old_id]
                new_grouped_indices[old_group].extend(new_ids)
        else:
            for new_id in new_indices:
                old_group = id2group[new_id]
                new_grouped_indices[old_group].append(new_id)
        self.grouped_indices = {
            k: new_grouped_indices[k] for k in sorted(new_grouped_indices.keys())
        }
        # logger.warning(f"id2group: {id2group}\n\n"
        #                f"new_indices: {new_indices}\n\n"
        #                f"self.grouped_indices: {self.grouped_indices}")
        new_indices = sum(new_grouped_indices.values(), [])
        self.indices = sorted(new_indices)


class AnalyzedDataset(AnalyzedData, Dataset):
    pass


class GroupedSlicedDataset(GroupedDataset, SlicedDataset):
    assert_types = ["SlicedDataset", "GroupedDataset"]
    excluded_types = ["AnalyzedData"]
    type_mapping = {
        (
            "SlicedDataset",
            "GroupedDataset",
            "GroupedSlicedDataset",
        ): "GroupedSlicedDataset",
    }

    def get_slice_info(self, ignore_groups=False) -> pd.DataFrame:
        """Concatenates slice_info Series and returns them as a DataFrame."""
        if ignore_groups or not self.is_grouped:
            concatenated_info = pd.concat(
                self.slice_info.values(), keys=self.slice_info.keys(), axis=1
            ).T
            concatenated_info.index.rename(self.index_levels["indices"], inplace=True)
            return concatenated_info
        else:
            group_dfs = {}
            for group, index_group in self.iter_grouped_indices():
                group_info = {ix: self.slice_info[ix] for ix in index_group}
                group_dfs[group] = pd.concat(
                    group_info.values(), keys=group_info.keys(), axis=1
                ).T
            concatenated_info = pd.concat(group_dfs.values(), keys=group_dfs.keys())
            concatenated_info.index = self._rename_multiindex_levels(
                concatenated_info.index,
                self.index_levels["groups"] + self.index_levels["indices"],
            )
            return clean_index_levels(concatenated_info)

    # def iter_facet(self, what, unfold=False, concatenate=False, ignore_groups=False):
    #     """Iterate through groups of potentially sliced facet DataFrames.
    #
    #     Parameters
    #     ----------
    #     what : {'form_labels', 'events', 'expanded', 'notes_and_rests', 'notes', 'labels',
    #             'cadences', 'chords', 'measures', 'rests'}
    #         What facet to retrieve.
    #     unfold : :obj:`bool`, optional
    #         Pass True if you need repeats to be unfolded.
    #     concatenate : :obj:`bool`, optional
    #         By default, the returned dict contains one DataFrame per ID in the group.
    #         Pass True to instead concatenate the DataFrames. Then, the dict will contain only
    #         one entry where the key is a tuple containing all IDs and the value is a DataFrame,
    #         the components of which can be distinguished using its MultiIndex.
    #     ignore_groups : :obj:`bool`, False
    #         If set to True, the iteration loop is flattened and yields (index, facet_df) pairs directly. Clashes
    #         with the setting concatenate=True which concatenates facets per group.
    #
    #     Yields
    #     ------
    #     :obj:`tuple`
    #         Group identifier
    #     :obj:`dict` or :obj:`pandas.DataFrame`
    #         Default: {ID -> DataFrame}.
    #         If concatenate=True: DataFrame with MultiIndex identifying ID, and (eventual) interval.
    #     """
    #     if not self.slice_facet_if_necessary(what, unfold):
    #         logger.info(f"No sliced {what} available.")
    #         raise StopIteration
    #     if sum((concatenate, ignore_groups)) > 1:
    #         raise ValueError(
    #             "Arguments 'concatenate' and 'ignore_groups' are in conflict, choose one "
    #             "or use the method get_facet()."
    #         )
    #     for group, index_group in self.iter_grouped_indices():
    #         result = {}
    #         missing_id = []
    #         for index in index_group:
    #             df = self.get_item(index, what=what, unfold=unfold)
    #             if df is None:
    #                 continue
    #             elif ignore_groups:
    #                 yield index, df
    #             if len(df.index) == 0:
    #                 missing_id.append(index)
    #             result[index] = df
    #         if ignore_groups:
    #             continue
    #         n_results = len(result)
    #         if len(missing_id) > 0:
    #             if n_results == 0:
    #                 pass
    #                 # logger.info(f"No '{what}' available for {group}.")
    #             else:
    #                 logger.info(
    #                     f"Group {group} is missing '{what}' for the following indices:\n"
    #                     f"{missing_id}"
    #                 )
    #         if n_results == 0:
    #             continue
    #         if concatenate:
    #             if n_results == 1:
    #                 # workaround necessary because of nasty "cannot handle overlapping indices;
    #                 # use IntervalIndex.get_indexer_non_unique" error
    #                 result["empty"] = pd.DataFrame()
    #             result = pd.concat(
    #                 result.values(),
    #                 keys=result.keys(),
    #                 names=self.index_levels["indices"] + ["interval"],
    #             )
    #             result = {tuple(index_group): result}
    #
    #         yield group, result


class AnalyzedGroupedDataset(AnalyzedDataset, GroupedDataset):
    assert_types = ["GroupedDataset", "AnalyzedDataset"]
    type_mapping = {
        (
            "AnalyzedGroupedDataset",
            "AnalyzedDataset",
            "GroupedDataset",
        ): "AnalyzedGroupedDataset",
    }


class AnalyzedSlicedDataset(AnalyzedDataset, SlicedDataset):
    assert_types = ["SlicedDataset", "AnalyzedDataset"]
    excluded_types = []
    type_mapping = {
        (
            "AnalyzedSlicedDataset",
            "AnalyzedDataset",
            "SlicedDataset",
        ): "AnalyzedSlicedDataset",
    }
    pass


class AnalyzedGroupedSlicedDataset(AnalyzedSlicedDataset, GroupedSlicedDataset):
    assert_types = [
        "GroupedSlicedDataset",
        "AnalyzedGroupedDataset",
        "AnalyzedSlicedDataset",
    ]
    type_mapping = {
        (
            "GroupedSlicedDataset",
            "AnalyzedGroupedDataset",
            "AnalyzedSlicedDataset",
            "AnalyzedGroupedSlicedDataset",
        ): "AnalyzedGroupedSlicedDataset",
    }
    pass


def remove_corpus_from_ids(result):
    """Called when group contains corpus_names and removes redundant repetition from indices."""
    if isinstance(result, dict):
        without_corpus = {}
        for key, v in result.items():
            if isinstance(key[0], str):
                without_corpus[key[1:]] = v
            else:
                new_key = tuple(k[1:] for k in key)
                without_corpus[new_key] = v
        return without_corpus
    logger.info(result)
    return result.droplevel(0)
