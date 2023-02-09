"""Class hierarchy for data types."""
import logging
from abc import ABC, abstractmethod
from typing import Any, Collection, Dict, Iterator, List, Optional, Tuple, Union

import pandas as pd

from dimcat._typing import ID, GroupID, SliceID
from dimcat.base import Data
from dimcat.utils.functions import typestrings2types

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

    def set_indices(self, new_indices: List[tuple]) -> None:
        """Replace :attr:`indices` with a new list of IDs.

        Args:
            new_indices:
                The new list IDs or a dictionary of several lists of IDs. The latter is useful for re-grouping
                freshly sliced IDs of a :class:`GroupedDataset`.
        """
        if isinstance(new_indices, dict):
            new_indices = sum(new_indices.values(), [])
        self.indices = new_indices

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

    def __new__(
        cls, data: Data, **kwargs
    ) -> Union["SlicedDataset", "GroupedSlicedDataset"]:
        return super().__new__(cls, data=data, **kwargs)

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

    def iter_slice_info(self) -> Iterator[Tuple[SliceID, pd.Series]]:
        """Iterate through concatenated slice_info Series for each group."""
        yield from self.slice_info.items()


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

    def __new__(
        cls, data: Data, **kwargs
    ) -> Union[
        "GroupedDataset",
        "GroupedSlicedDataset",
        "AnalyzedGroupedDataset",
        "AnalyzedGroupedSlicedDataset",
    ]:
        return super().__new__(cls, data=data, **kwargs)

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

    def iter_grouped_indices(self) -> Iterator[Tuple[str, List[ID]]]:
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

    def iter_grouped_slice_info(self) -> Iterator[Tuple[tuple, pd.DataFrame]]:
        """Iterate through concatenated slice_info DataFrame for each group."""
        for group, index_group in self.iter_grouped_indices():
            group_info = {ix: self.slice_info[ix] for ix in index_group}
            group_df = pd.concat(group_info.values(), keys=group_info.keys(), axis=1).T
            group_df.index = self._rename_multiindex_levels(
                group_df.index, self.index_levels["indices"]
            )
            yield group, group_df


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

    def __new__(
        cls, data: Data, **kwargs
    ) -> Union[
        "AnalyzedDataset",
        "AnalyzedGroupedDataset",
        "AnalyzedSlicedDataset",
        "AnalyzedGroupedSlicedDataset",
    ]:
        return super().__new__(cls, data=data, **kwargs)

    def __init__(self, data: Data, **kwargs):
        logger.debug(f"{type(self).__name__} -> before {super()}.__init__()")
        super().__init__(data=data, **kwargs)
        logger.debug(f"{type(self).__name__} -> after {super()}.__init__()")
        if not hasattr(self, "processed"):
            if hasattr(data, "processed"):
                self.processed = data.processed
            else:
                self.processed: List["Result"] = []
                """Analyzers store there result here using :meth:`set_result`."""

    def set_result(self, analyzer: "Analyzer", result: "Result"):  # noqa: F821
        assert analyzer == self.get_previous_pipeline_step()
        self.processed = [result] + self.processed

    # @overload
    # def get(self, as_pandas: bool = Literal[True]) -> Pandas:
    #     ...
    #
    # @overload
    # def get(self, as_pandas: bool = Literal[False]) -> Dict[GroupID, Any]:
    #     ...
    #
    # def get(self, as_pandas: bool = True) -> Union[Pandas, Dict[GroupID, Any]]:
    #     """Collects the results of :meth:`iter` to retrieve all processed data at once.
    #
    #     Args:
    #         as_pandas:
    #             By default, the result is a pandas DataFrame or Series where the first levels
    #             display group identifiers (if any). Pass False to obtain a nested {group -> group_result}
    #             dictionary instead.
    #
    #     Returns:
    #         The contents of :attr:`processed` in original or adapted form.
    #     """
    #     if len(self.processed) == 0:
    #         logger.info("No data has been processed so far.")
    #         return
    #     results = {group: result for group, result in self.iter(as_pandas=as_pandas)}
    #     if not as_pandas:
    #         return results
    #     if self.group2pandas is None:
    #         return pd.Series(results)
    #     # default: concatenate to a single pandas object
    #     if len(results) == 1 and () in results:
    #         pandas_obj = pd.concat(results.values())
    #     else:
    #         try:
    #             pandas_obj = pd.concat(
    #                 results.values(),
    #                 keys=results.keys(),
    #                 names=self.index_levels["groups"],
    #             )
    #         except ValueError:
    #             logger.info(self.index_levels["groups"])
    #             logger.info(results.keys())
    #             raise
    #     return clean_index_levels(pandas_obj)
    #
    def get_result_object(self, idx=0):
        return self.processed[idx]

    def get_results(self) -> pd.DataFrame:
        result_obj = self.get_result_object()
        return result_obj.get_results()

    def get_group_results(self) -> pd.DataFrame:
        result_obj = self.get_result_object()
        return result_obj.get_group_results()

    def iter_results(self):
        result_obj = self.get_result_object()
        yield from result_obj.iter_results()

    def iter_group_results(self):
        result_obj = self.get_result_object()
        yield from result_obj.iter_group_results()

    # @overload
    # def iter(
    #     self, as_pandas: bool = Literal[False], ignore_groups: bool = Literal[False]
    # ) -> Iterator[Tuple[GroupID, Union[Dict[ID, Any], Any]]]:
    #     ...
    #
    # @overload
    # def iter(
    #     self, as_pandas: bool = Literal[True], ignore_groups: bool = Literal[False]
    # ) -> Iterator[Tuple[GroupID, Union[Pandas, Any]]]:
    #     ...
    #
    # @overload
    # def iter(
    #     self, as_pandas: bool = Literal[False], ignore_groups: bool = Literal[True]
    # ) -> Iterator[Union[Tuple[ID, Any], Any]]:
    #     ...
    #
    # @overload
    # def iter(
    #     self, as_pandas: bool = Literal[True], ignore_groups: bool = Literal[True]
    # ) -> Iterator[Union[Pandas, Any]]:
    #     ...
    #
    # def iter(
    #     self, as_pandas: bool = True, ignore_groups: bool = False
    # ) -> Iterator[
    #     Union[
    #         Tuple[GroupID, Union[Dict[ID, Any], Any]],
    #         Tuple[GroupID, Union[Pandas, Any]],
    #         Union[Tuple[ID, Any], Any],
    #         Union[Pandas, Any],
    #     ]
    # ]:
    #     """Iterate through :attr:`processed` data.
    #
    #     Args:
    #         as_pandas:
    #             Setting this value to False corresponds to iterating through .processed.items(),
    #             where keys are group IDs and values are results for Analyzers that compute
    #             one result per group, or {ID -> result} dicts for Analyzers that compute
    #             one result per item per group. The default value (True) has no effect in the first case,
    #             but in the second case, the dictionary will be converted to a Series if the conversion method is
    #             set in :attr:`group2pandas`.
    #         ignore_groups:
    #             If set to True, the iteration loop is flattened and does not include group identifiers. If as_pandas
    #             is False (default), and the applied Analyzer computes one {ID -> result} dict per group,
    #             this will correspond to iterating through the (ID, result) tuples for all groups.
    #
    #     Yields:
    #         The result of the last applied Analyzer for each group or for each item of each group.
    #     """
    #     if ignore_groups and not as_pandas:
    #         raise ValueError(
    #             "If you set 'as_dict' and 'ignore_groups' are in conflict, choose one or use _.get()."
    #         )
    #     for group, result in self.processed.items():
    #         if ignore_groups:
    #             if self.group2pandas is None:
    #                 yield result
    #             elif as_pandas:
    #                 yield self.convert_group2pandas(result)
    #             else:
    #                 yield from result.items()
    #         else:
    #             if as_pandas and self.group2pandas is not None:
    #                 yield group, self.convert_group2pandas(result)
    #             else:
    #                 yield group, result


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
